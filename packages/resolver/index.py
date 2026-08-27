"""
The brand index — 253,974 Indian medicines, searchable from corrupted OCR.

Two structural decisions, both driven by what the old implementation got wrong.

**Built once, loaded as an artifact.** The previous version
(`ml-service/medicine_db.py`) read every row into a `list[dict]` and re-fitted a
TF-IDF vectoriser at process start. A quarter-million Python dicts is roughly
250 MB of interpreter objects before any model exists, and re-fitting on every
boot makes startup time a function of dataset size. Here the vectoriser and
matrix are fitted by `scripts/build_index.py` and persisted; the service
memory-maps columnar arrays and loads a pre-fitted matrix. This is what makes a
2 GB instance and a free-tier database viable — an infrastructure constraint
solved by data layout rather than by paying for more RAM.

**Character n-grams, not words.** OCR of a torn strip produces `AUGMENTlN`,
`Augrnentin`, `UGMENTIN`. Word-level matching scores all three at zero against
`augmentin` because no whole token matches. Character 2-4 grams degrade
gracefully instead: `Augrnentin` still shares `aug`, `ugm`, `nti`, `tin` with
the target, so similarity falls rather than collapsing. `char_wb` keeps n-grams
inside word boundaries, which stops spurious grams forming across the gap
between two unrelated words.

Similarity is a plain sparse matrix-vector product rather than sklearn's
`NearestNeighbors`. With L2-normalised TF-IDF rows, cosine similarity *is* the
dot product, so the wrapper adds an abstraction and a distance-to-similarity
conversion without adding anything.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from .normalize import (
    Ingredient,
    PackSize,
    brand_root,
    composition_signature,
    normalize_text,
    parse_bracketed_composition,
    parse_dosage_form,
    parse_inline_composition,
    parse_pack_size,
    parse_price,
    parse_unit_size,
    signature_string,
)

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

ARTIFACT_VERSION = 4
"""Bumped whenever the artifact layout changes, so a stale index is detected at
load time rather than producing silently wrong matches."""

AGREEMENT_BONUS = 0.25
"""Weight on the weaker of the two field similarities.

Scores fuse as `max(name, composition) + AGREEMENT_BONUS * min(name, composition)`
— a soft OR with a corroboration term. Either field alone can carry a match,
and agreement between them adds a modest bonus.

Chosen by measuring four fusions on 1,200 corrupted queries (composition
accuracy, top-1 / top-5):

    fusion                  name only        name + composition
    weighted sum w=0.5      0.583 / 0.699    0.882 / 0.971
    weighted sum w=0.65     0.601 / 0.714    0.838 / 0.954
    max                     0.593 / 0.705    0.892 / 0.954
    max + 0.25*min          0.600 / 0.711    0.927 / 0.983

A weighted sum is the obvious choice and the wrong one: it lets a mediocre
score in one field dilute a strong score in the other. `crocin 500` returned
azithromycin under a weighted sum, because "500" matched dozens of
`...500mg` compositions well enough to outweigh a decisive brand-name hit on
Crocin. A max-based rule cannot be dragged down that way."""

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "data" / "artifacts"


def _make_vectorizer() -> TfidfVectorizer:
    """Character n-gram vectoriser, shared by both fields.

    `char_wb` 2-4 grams: character n-grams degrade gracefully under OCR damage
    where word matching collapses. `Augrnentin` shares `aug`, `ugm`, `nti`,
    `tin` with `augmentin`, so similarity falls rather than going to zero. The
    `_wb` variant keeps n-grams inside word boundaries, preventing spurious
    grams forming across the gap between two unrelated words.
    """
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=2,
        max_features=200_000,
        dtype=np.float32,
        sublinear_tf=True,
    )


def _token_query(tokens: Iterable[str], strengths: Iterable[tuple[float, str]]) -> str:
    parts = [t for t in tokens if t]
    parts.extend(f"{value:g}{unit}" for value, unit in strengths)
    return " ".join(parts)


@dataclass
class CompositionMatch:
    """A candidate composition, with the evidence supporting it.

    This is the resolver's primary unit of output. See
    `BrandIndex.search_compositions` for why composition rather than brand.
    """

    signature: tuple
    label: str
    best_row: int
    best_name: str
    top_similarity: float
    aggregate_score: float
    support: int
    """How many distinct brands in the candidate pool share this composition."""

    example_rows: list[int] = field(default_factory=list)

    dense_similarity: float | None = None
    """Cosine similarity from the dense (embedding) index, when fusion ran.
    None means the lexical stage alone produced this ranking."""

    fused_rank_score: float | None = None
    """Reciprocal-rank-fusion score, when fusion ran. See resolver/dense.py."""

    def to_dict(self) -> dict:
        payload = {
            "composition": self.label,
            "closest_brand": self.best_name,
            "top_similarity": round(self.top_similarity, 4),
            "aggregate_score": round(self.aggregate_score, 4),
            "supporting_brands": self.support,
        }
        if self.dense_similarity is not None:
            payload["dense_similarity"] = round(self.dense_similarity, 4)
        return payload


@dataclass
class BrandRecord:
    """One branded medicine, with everything the pharmacology layer needs."""

    row: int
    name: str
    manufacturer: str
    price: float | None
    pack_count: float | None
    pack_unit: str
    pack_label: str
    dosage_form: str | None
    composition: str
    signature: tuple
    discontinued: bool
    similarity: float = 0.0
    name_similarity: float = 0.0
    composition_similarity: float = 0.0

    @property
    def price_per_unit(self) -> float | None:
        """Rupees per tablet / per ml / per gram, whichever the pack is in.

        The unit travels with the number everywhere downstream — comparing a
        per-ml price against a per-tablet price is the arithmetic error the
        whole affordability feature exists to avoid making.
        """
        if self.price is None or not self.pack_count:
            return None
        return round(self.price / self.pack_count, 4)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "manufacturer": self.manufacturer,
            "price": self.price,
            "pack": {"count": self.pack_count, "unit": self.pack_unit, "label": self.pack_label},
            "price_per_unit": self.price_per_unit,
            "dosage_form": self.dosage_form,
            "composition": self.composition,
            "discontinued": self.discontinued,
            "similarity": round(self.similarity, 4),
            "name_similarity": round(self.name_similarity, 4),
            "composition_similarity": round(self.composition_similarity, 4),
            "source": {"dataset": "a_z_medicines_india", "record_id": str(self.row)},
        }


@dataclass
class GenericRecord:
    """One Jan Aushadhi (PMBJP) product."""

    row: int
    name: str
    drug_code: str
    mrp: float | None
    pack_count: float | None
    pack_unit: str
    unit_size: str
    category: str
    dosage_form: str | None
    signature: tuple

    @property
    def price_per_unit(self) -> float | None:
        if self.mrp is None or not self.pack_count:
            return None
        return round(self.mrp / self.pack_count, 4)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "drug_code": self.drug_code,
            "mrp": self.mrp,
            "pack": {"count": self.pack_count, "unit": self.pack_unit, "label": self.unit_size},
            "price_per_unit": self.price_per_unit,
            "category": self.category,
            "dosage_form": self.dosage_form,
            "source": {
                "dataset": "jan_aushadhi_pmbjp",
                "record_id": self.drug_code or str(self.row),
                "url": "https://janaushadhi.gov.in",
            },
        }


# --- building -------------------------------------------------------------


def _read_csv(path: Path, encoding: str = "utf-8") -> list[dict]:
    with path.open(newline="", encoding=encoding) as handle:
        return list(csv.DictReader(handle))


def build(
    data_dir: Path = DEFAULT_DATA_DIR,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    *,
    verbose: bool = True,
) -> dict:
    """Build and persist the index artifacts. Run by `scripts/build_index.py`."""

    def log(message: str) -> None:
        if verbose:
            print(f"[build_index] {message}", flush=True)

    artifact_dir.mkdir(parents=True, exist_ok=True)

    # --- brands ---
    log("reading A_Z brands...")
    brand_rows = _read_csv(data_dir / "A_Z_medicines_dataset_of_India.csv")
    log(f"  {len(brand_rows):,} rows")

    names: list[str] = []
    roots: list[str] = []
    manufacturers: list[str] = []
    prices: list[float] = []
    pack_counts: list[float] = []
    pack_units: list[str] = []
    pack_labels: list[str] = []
    forms: list[str] = []
    compositions: list[str] = []
    discontinued: list[bool] = []
    signatures: list[tuple] = []

    for row in brand_rows:
        name = (row.get("name") or "").strip()
        ingredients = parse_bracketed_composition(
            row.get("short_composition1", ""), row.get("short_composition2", "")
        )
        pack = parse_pack_size(row.get("pack_size_label", ""))

        names.append(name)
        roots.append(brand_root(name))
        manufacturers.append((row.get("manufacturer_name") or "").strip())
        price = parse_price(row.get("price(₹)") or row.get("price"))
        prices.append(np.nan if price is None else price)
        pack_counts.append(np.nan if pack.count is None else pack.count)
        pack_units.append(pack.unit)
        pack_labels.append(pack.raw)
        forms.append(parse_dosage_form(row.get("pack_size_label", "")) or "")
        compositions.append(signature_string(ingredients))
        discontinued.append(str(row.get("Is_discontinued", "")).strip().upper() == "TRUE")
        signatures.append(composition_signature(ingredients))

    # Two separate matrices, not one concatenated corpus.
    #
    # Measured on 1,200 corrupted queries, concatenating brand root and
    # composition into one string scored 0.296 top-1 — the *worst* of four
    # layouts tried, and well below matching on the name alone (0.519). Mixing
    # them dilutes both: composition text is shared by ~24 brands on average, so
    # its n-grams swamp the handful that actually distinguish one brand from
    # another.
    #
    # Kept apart, the two fields answer different questions and their scores can
    # be weighted per query. That matters because they are readable under
    # different damage: a stylised brand logo is often the first thing glare
    # destroys, while the composition line is set in plain small type and
    # frequently survives.
    log("fitting name vectoriser (char_wb 2-4 grams)...")
    name_corpus = [normalize_text(n) for n in names]
    name_vectorizer = _make_vectorizer()
    name_matrix = name_vectorizer.fit_transform(name_corpus)
    log(f"  name matrix {name_matrix.shape}, {name_matrix.nnz:,} nnz, "
        f"{name_matrix.data.nbytes / 1e6:.1f} MB")

    log("fitting composition vectoriser...")
    composition_corpus = [c if c else "unknown" for c in compositions]
    composition_vectorizer = _make_vectorizer()
    composition_matrix = composition_vectorizer.fit_transform(composition_corpus)
    log(f"  composition matrix {composition_matrix.shape}, "
        f"{composition_matrix.nnz:,} nnz, {composition_matrix.data.nbytes / 1e6:.1f} MB")

    # --- generics (Jan Aushadhi) ---
    log("reading Jan Aushadhi catalogue...")
    generic_rows = _read_csv(data_dir / "generic.csv", encoding="utf-8-sig")
    generics: list[GenericRecord] = []
    for i, row in enumerate(generic_rows):
        gname = (row.get("Generic Name") or "").strip()
        ingredients = parse_inline_composition(gname)
        pack = parse_unit_size(row.get("Unit Size", ""))
        generics.append(
            GenericRecord(
                row=i,
                name=gname,
                drug_code=(row.get("Drug Code") or "").strip(),
                mrp=parse_price(row.get("MRP")),
                pack_count=pack.count,
                pack_unit=pack.unit,
                unit_size=pack.raw,
                category=(row.get("Group Name") or "").strip(),
                dosage_form=parse_dosage_form(gname),
                signature=composition_signature(ingredients),
            )
        )
    log(f"  {len(generics):,} rows")

    # --- inverted indexes on composition signature ---
    # This is what turns "find a cheaper equivalent" into a dictionary lookup
    # over products proven to share a composition, rather than a similarity
    # search that might return a different drug that merely sounds alike.
    log("building signature indexes...")
    brand_by_signature: dict[tuple, list[int]] = defaultdict(list)
    for i, signature in enumerate(signatures):
        if signature:
            brand_by_signature[signature].append(i)

    generic_by_signature: dict[tuple, list[int]] = defaultdict(list)
    for generic in generics:
        if generic.signature:
            generic_by_signature[generic.signature].append(generic.row)

    log(f"  {len(brand_by_signature):,} distinct brand signatures")
    log(f"  {len(generic_by_signature):,} distinct generic signatures")
    shared = set(brand_by_signature) & set(generic_by_signature)
    log(f"  {len(shared):,} signatures present in BOTH -> substitutable today")

    columns = {
        "name": np.array(names, dtype=object),
        "root": np.array(roots, dtype=object),
        "manufacturer": np.array(manufacturers, dtype=object),
        "price": np.asarray(prices, dtype=np.float32),
        "pack_count": np.asarray(pack_counts, dtype=np.float32),
        "pack_unit": np.array(pack_units, dtype=object),
        "pack_label": np.array(pack_labels, dtype=object),
        "form": np.array(forms, dtype=object),
        "composition": np.array(compositions, dtype=object),
        "discontinued": np.asarray(discontinued, dtype=bool),
    }

    log("writing artifacts...")
    joblib.dump(name_vectorizer, artifact_dir / "name_vectorizer.joblib", compress=3)
    joblib.dump(composition_vectorizer, artifact_dir / "composition_vectorizer.joblib", compress=3)
    sp.save_npz(artifact_dir / "name_matrix.npz", name_matrix)
    sp.save_npz(artifact_dir / "composition_matrix.npz", composition_matrix)
    joblib.dump(
        {
            "version": ARTIFACT_VERSION,
            "columns": columns,
            "signatures": signatures,
            "brand_by_signature": dict(brand_by_signature),
            "generics": generics,
            "generic_by_signature": dict(generic_by_signature),
        },
        artifact_dir / "index_meta.joblib",
        compress=3,
    )

    stats = {
        "brands": len(names),
        "generics": len(generics),
        "name_features": int(name_matrix.shape[1]),
        "composition_features": int(composition_matrix.shape[1]),
        "brand_signatures": len(brand_by_signature),
        "generic_signatures": len(generic_by_signature),
        "substitutable_signatures": len(shared),
        "brands_per_signature": round(len(names) / max(len(brand_by_signature), 1), 1),
        "matrix_mb": round(
            (name_matrix.data.nbytes + composition_matrix.data.nbytes) / 1e6, 1
        ),
    }
    log(f"done: {stats}")
    return stats


# --- loading and searching ------------------------------------------------


class BrandIndex:
    """Loaded index. Construct once per process and share it."""

    def __init__(self, artifact_dir: Path = DEFAULT_ARTIFACT_DIR):
        self.artifact_dir = Path(artifact_dir)
        meta_path = self.artifact_dir / "index_meta.joblib"

        if not meta_path.exists():
            raise FileNotFoundError(
                f"index artifacts not found in {self.artifact_dir}. "
                "Run: python scripts/build_index.py"
            )

        meta = joblib.load(meta_path)
        if meta.get("version") != ARTIFACT_VERSION:
            raise RuntimeError(
                f"index artifact version {meta.get('version')} != expected "
                f"{ARTIFACT_VERSION}. Rebuild with: python scripts/build_index.py"
            )

        self._columns = meta["columns"]
        self._signatures: list[tuple] = meta["signatures"]
        self._brand_by_signature: dict[tuple, list[int]] = meta["brand_by_signature"]
        self._generics: list[GenericRecord] = meta["generics"]
        self._generic_by_signature: dict[tuple, list[int]] = meta["generic_by_signature"]

        self._name_vectorizer: TfidfVectorizer = joblib.load(
            self.artifact_dir / "name_vectorizer.joblib"
        )
        self._composition_vectorizer: TfidfVectorizer = joblib.load(
            self.artifact_dir / "composition_vectorizer.joblib"
        )
        self._name_matrix = sp.load_npz(self.artifact_dir / "name_matrix.npz").tocsr()
        self._composition_matrix = sp.load_npz(
            self.artifact_dir / "composition_matrix.npz"
        ).tocsr()

    def __len__(self) -> int:
        return len(self._signatures)

    @property
    def generic_count(self) -> int:
        return len(self._generics)

    def record(self, row: int, similarity: float = 0.0) -> BrandRecord:
        """Materialise one row into a `BrandRecord`.

        Rows are built on demand rather than held as objects. Only the handful
        of candidates a query actually returns get instantiated, which is what
        keeps a quarter-million-row index off the heap.
        """
        columns = self._columns
        price = float(columns["price"][row])
        pack_count = float(columns["pack_count"][row])

        return BrandRecord(
            row=row,
            name=columns["name"][row],
            manufacturer=columns["manufacturer"][row],
            price=None if np.isnan(price) else price,
            pack_count=None if np.isnan(pack_count) else pack_count,
            pack_unit=columns["pack_unit"][row],
            pack_label=columns["pack_label"][row],
            dosage_form=columns["form"][row] or None,
            composition=columns["composition"][row],
            signature=self._signatures[row],
            discontinued=bool(columns["discontinued"][row]),
            similarity=similarity,
        )

    def _score(self, query: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-row (fused, name, composition) similarity against `query`."""
        normalised = normalize_text(query)
        if not normalised:
            zeros = np.zeros(len(self), dtype=np.float32)
            return zeros, zeros, zeros

        # Rows are L2-normalised by TfidfVectorizer, so the dot product is
        # exactly cosine similarity — no separate normalisation step.
        name_scores = (
            self._name_matrix @ self._name_vectorizer.transform([normalised]).T
        ).toarray().ravel()
        composition_scores = (
            self._composition_matrix
            @ self._composition_vectorizer.transform([normalised]).T
        ).toarray().ravel()

        fused = np.maximum(name_scores, composition_scores) + AGREEMENT_BONUS * np.minimum(
            name_scores, composition_scores
        )
        return fused, name_scores, composition_scores

    def search(
        self,
        query: str,
        *,
        top_k: int = 25,
        min_similarity: float = 0.10,
        include_discontinued: bool = False,
    ) -> list[BrandRecord]:
        """Lexical search, fusing brand-name and composition similarity.

        Returns candidates ranked by the fused score. Deliberately a *set*, not
        a best guess: choosing among them, and deciding whether to answer at
        all, belongs to the calibration stage, which has more evidence than a
        similarity score alone.
        """
        scores, name_scores, composition_scores = self._score(query)
        if scores.max(initial=0.0) <= 0:
            return []

        # argpartition finds the top k in O(n) rather than sorting all 253,973
        # scores. Only the retained values are then sorted.
        count = min(max(top_k * 4, 32), scores.size)
        candidate_rows = np.argpartition(-scores, count - 1)[:count]
        candidate_rows = candidate_rows[np.argsort(-scores[candidate_rows])]

        results: list[BrandRecord] = []
        for row in candidate_rows:
            similarity = float(scores[row])
            if similarity < min_similarity:
                break
            if not include_discontinued and self._columns["discontinued"][row]:
                continue
            record = self.record(int(row), similarity)
            record.name_similarity = float(name_scores[row])
            record.composition_similarity = float(composition_scores[row])
            results.append(record)
            if len(results) >= top_k:
                break

        return results

    def search_compositions(
        self,
        query: str,
        *,
        top_k: int = 5,
        pool: int = 200,
    ) -> list[CompositionMatch]:
        """Rank *compositions*, not brand rows. This is the resolver's real output.

        253,973 brands share 10,780 compositions — about 24 brands each — so
        top-1 brand-row accuracy is arithmetically capped near 1/24 whenever the
        brand name itself is unreadable. Measured against corrupted queries,
        brand-row accuracy was 0.35 while composition accuracy on the same
        queries was 0.92. The first number looks like a broken system; the
        second describes the same system working well.

        Composition is also the answer the rest of the pipeline needs. Drug
        interactions, NPPA ceiling prices, schedule classification and generic
        substitution are all properties of the molecule and its strength.
        Whether the strip is Augmentin 625 Duo or any of the other twenty-three
        amoxycillin-500/clavulanate-125 tablets changes the label on the box and
        nothing clinically relevant.

        Evidence is aggregated across every candidate row sharing a signature,
        so a composition supported by several independent near-matches
        outranks one supported by a single lucky hit.
        """
        scores, _, _ = self._score(query)
        if scores.max(initial=0.0) <= 0:
            return []

        count = min(max(pool, top_k * 20), scores.size)
        candidate_rows = np.argpartition(-scores, count - 1)[:count]
        candidate_rows = candidate_rows[np.argsort(-scores[candidate_rows])]

        grouped: dict[tuple, list[tuple[int, float]]] = defaultdict(list)
        for row in candidate_rows:
            signature = self._signatures[row]
            if signature:
                grouped[signature].append((int(row), float(scores[row])))

        matches: list[CompositionMatch] = []
        for signature, hits in grouped.items():
            # Ranked by the single best-matching row, NOT by an aggregate over
            # supporters. An earlier version multiplied the score by a support
            # bonus, on the theory that several near-matches corroborate each
            # other. They do not: brands sharing a composition are the same
            # composition listed many times, not independent evidence. All the
            # bonus did was reward market share — `crocin 500` returned
            # azithromycin (39 brands) over paracetamol (2 brands) despite
            # paracetamol having the better lexical match.
            #
            # Support is still reported, because it is genuinely useful context
            # for the user ("39 brands sell this exact composition"). It just
            # does not decide the ranking.
            matches.append(
                CompositionMatch(
                    signature=signature,
                    label=self._columns["composition"][hits[0][0]],
                    best_row=hits[0][0],
                    best_name=self._columns["name"][hits[0][0]],
                    top_similarity=hits[0][1],
                    aggregate_score=round(hits[0][1], 5),
                    support=len(hits),
                    example_rows=[row for row, _ in hits[:5]],
                )
            )

        matches.sort(key=lambda m: m.top_similarity, reverse=True)
        return matches[:top_k]

    def search_tokens(
        self, tokens: Iterable[str], *, strengths: Iterable[tuple[float, str]] = (), **kwargs
    ) -> list[BrandRecord]:
        """Search from an OCR token bag rather than a clean name."""
        return self.search(_token_query(tokens, strengths), **kwargs)

    def search_compositions_from_tokens(
        self, tokens: Iterable[str], *, strengths: Iterable[tuple[float, str]] = (), **kwargs
    ) -> list[CompositionMatch]:
        """Composition ranking from an OCR token bag. The main scan path.

        Parsed dose strengths are appended to the query. They are highly
        diagnostic and survive damage that destroys brand names: `500mg` with
        `125mg` pins an amoxycillin/clavulanate co-formulation even when every
        letter of "Augmentin" is unreadable, because very few products share
        that exact pair.
        """
        return self.search_compositions(_token_query(tokens, strengths), **kwargs)

    def by_signature(self, signature: tuple, *, limit: int | None = None) -> list[BrandRecord]:
        """Every brand sharing an exact composition signature."""
        rows = self._brand_by_signature.get(signature, [])
        if limit is not None:
            rows = rows[:limit]
        return [self.record(row) for row in rows]

    def generics_by_signature(self, signature: tuple) -> list[GenericRecord]:
        """Every Jan Aushadhi product sharing an exact composition signature."""
        return [self._generics[row] for row in self._generic_by_signature.get(signature, [])]

    def all_generics(self) -> list[GenericRecord]:
        return list(self._generics)

    def discriminative_vocabulary(self, max_document_frequency: float = 0.02) -> set[str]:
        """Ingredient words that actually identify something.

        A word is kept only if it appears in fewer than `max_document_frequency`
        of distinct compositions. That threshold is what separates `belladonna`
        (one composition family) from `and` (thousands) — both are technically
        present in the composition vocabulary, but only one narrows anything
        down.

        Used to protect real ingredient names from boilerplate filtering without
        also protecting the connectors inside multi-word names.
        """
        from collections import Counter

        seen: set[tuple] = set()
        counts: Counter = Counter()

        for row, signature in enumerate(self._signatures):
            if not signature or signature in seen:
                continue
            seen.add(signature)
            words = {
                w
                for w in str(self._columns["composition"][row]).replace("+", " ").split()
                if w.isalpha() and len(w) > 2
            }
            counts.update(words)

        total = max(len(seen), 1)
        return {w for w, n in counts.items() if n / total < max_document_frequency}

    def stats(self) -> dict:
        return {
            "brands": len(self),
            "generics": self.generic_count,
            "brand_signatures": len(self._brand_by_signature),
            "generic_signatures": len(self._generic_by_signature),
            "name_features": int(self._name_matrix.shape[1]),
            "composition_features": int(self._composition_matrix.shape[1]),
        }


_INDEX: BrandIndex | None = None


def get_index(artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> BrandIndex:
    """Process-wide singleton. Loading is expensive; sharing it is the point."""
    global _INDEX
    if _INDEX is None:
        _INDEX = BrandIndex(artifact_dir)
    return _INDEX
