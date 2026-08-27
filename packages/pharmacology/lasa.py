"""
Look-alike / sound-alike (LASA) confusability detection.

Given a product, find OTHER products whose *names* are easily confused with it
but whose *composition differs*. That exclusion is the whole point: two
products sharing a composition are the same medicine under different labels,
which is substitution — useful, and already handled by
`pharmacology/alternatives.py`. Confusion is only dangerous when the names are
close and the contents are not.

No model is involved at any stage, in keeping with the rule that this package
contains no LLM calls. Candidates come from the existing lexical index; the
ranking is string arithmetic.

Thresholds are measured, not guessed. Over 4,000 random name pairs drawn from
the catalogue (seed 11):

    signal                       random median   random p99   threshold here
    normalised Damerau-Lev.          0.258         0.667          0.70
    Jaro-Winkler                     0.553         0.815          0.85
    metaphone equality                 --      0/4000 collisions  any match

Each threshold sits just above the 99th percentile of chance, so a flagged
pair is genuinely unusual rather than typical. Metaphone equality never once
occurred by chance across those 4,000 pairs, which is why it qualifies on its
own.

Names are compared as brand *roots* (`normalize.brand_root`), so
"Celebrex 200mg Capsule" is compared as "celebrex". Without that, shared
strength and dosage-form words inflate similarity between products that have
nothing else in common.
"""

from __future__ import annotations

from dataclasses import dataclass

import jellyfish

from packages.resolver.normalize import brand_root

DAMERAU_SIMILARITY_FLOOR = 0.70
JARO_WINKLER_FLOOR = 0.85
MIN_ROOT_LENGTH = 4
"""Below four characters, edit distance stops discriminating — nearly every
short root is one or two edits from many others, and metaphone codes collide
freely. Such names are skipped rather than reported as universally confusable."""

CANDIDATE_POOL = 200
"""Rows pulled from the lexical index before reranking. The index already
ranks by name similarity, so the confusable neighbours are concentrated at the
top; 200 is deep enough to survive the composition-exclusion filter."""


@dataclass
class Confusable:
    """One product whose name is confusable with the query's."""

    name: str
    composition: str
    manufacturer: str
    signature: tuple
    why: list[str]
    edit_distance: int
    damerau_similarity: float
    jaro_winkler: float
    phonetic_match: bool
    similarity: float
    source: dict

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "composition": self.composition,
            "manufacturer": self.manufacturer,
            "why": self.why,
            "edit_distance": self.edit_distance,
            "damerau_similarity": round(self.damerau_similarity, 3),
            "jaro_winkler": round(self.jaro_winkler, 3),
            "phonetic_match": self.phonetic_match,
            "similarity": round(self.similarity, 3),
            "source": self.source,
        }


@dataclass
class LasaResult:
    query: str
    query_root: str
    confusable: list[Confusable]
    message: str
    caution: str

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "query_root": self.query_root,
            "confusable": [c.to_dict() for c in self.confusable],
            "count": len(self.confusable),
            "message": self.message,
            "caution": self.caution,
        }


def damerau_similarity(a: str, b: str) -> float:
    """Edit distance normalised to [0, 1] by the longer string."""
    longest = max(len(a), len(b))
    if longest == 0:
        return 0.0
    return 1.0 - (jellyfish.damerau_levenshtein_distance(a, b) / longest)


def score(a: str, b: str) -> dict:
    """All three signals for one pair of brand roots."""
    return {
        "edit_distance": jellyfish.damerau_levenshtein_distance(a, b),
        "damerau_similarity": damerau_similarity(a, b),
        "jaro_winkler": jellyfish.jaro_winkler_similarity(a, b),
        "phonetic_match": bool(a and b and jellyfish.metaphone(a) == jellyfish.metaphone(b)),
    }


def same_brand_family(a: str, b: str) -> bool:
    """Are these two roots line extensions of one brand rather than rivals?

    Measured problem this solves: the first working version reported
    "Augmentin 625 Duo Tablet" as confusable with "Augmentin 375 Tablet" and
    "Augmentin Duo Oral Suspension" at similarity 0.94, and "Crocin Advance"
    with "Crocin 1000mg". Those are the same brand at different strengths.
    The exact-root check missed them because `brand_root` leaves a trailing
    qualifier behind — "augmentin" against "augmentin duo" — so the strings
    are near-identical without being equal.

    Confusing two strengths of one brand is a real dispensing hazard, but it
    is a *different* hazard with a different remedy ("check the strength"),
    and at these similarity scores it crowds genuine cross-brand look-alikes
    out of the list entirely. It belongs to the price/alternatives path, which
    already reasons about strengths, not here.
    """
    if a == b:
        return True
    first_a, first_b = a.split(" ")[0], b.split(" ")[0]
    if first_a == first_b:
        return True
    return a.startswith(b + " ") or b.startswith(a + " ")


def reasons(signals: dict) -> list[str]:
    """Which signals fired — reported so a user can judge the warning."""
    why = []
    if signals["damerau_similarity"] >= DAMERAU_SIMILARITY_FLOOR:
        why.append("spelling")
    if signals["jaro_winkler"] >= JARO_WINKLER_FLOOR:
        why.append("prefix")
    if signals["phonetic_match"]:
        why.append("sound")
    return why


CAUTION = (
    "These products have names that are easy to confuse but do not contain the "
    "same active ingredients. If you were handed one of these instead of what "
    "you expected, check the composition on the pack against your prescription "
    "before taking it."
)
"""Fixed text, not generated. The only thing that varies between responses is
which retrieved products are listed."""


def find_confusable(
    index,
    *,
    name: str,
    signature: tuple = (),
    exclude_row: int | None = None,
    limit: int = 10,
) -> LasaResult:
    """Products with confusable names and a different composition.

    An empty list is the common and correct outcome — most brand names have no
    close neighbour, and saying so is more useful than padding the list with
    weak matches.
    """
    root = brand_root(name)

    if len(root) < MIN_ROOT_LENGTH:
        return LasaResult(
            query=name,
            query_root=root,
            confusable=[],
            message=(
                f"'{name}' reduces to a brand root of fewer than "
                f"{MIN_ROOT_LENGTH} characters, which is too short to compare "
                "reliably. No confusability check was performed."
            ),
            caution="",
        )

    candidates = index.search(root, top_k=CANDIDATE_POOL, min_similarity=0.0)

    seen_roots: set[str] = {root}
    found: list[Confusable] = []

    for record in candidates:
        if exclude_row is not None and record.row == exclude_row:
            continue
        # The exclusion that defines this feature: same composition means the
        # same medicine, which is substitution rather than confusion.
        if signature and record.signature == signature:
            continue

        other = brand_root(record.name)
        if len(other) < MIN_ROOT_LENGTH or other in seen_roots:
            continue
        if same_brand_family(root, other):
            continue

        signals = score(root, other)
        why = reasons(signals)
        if not why:
            continue

        seen_roots.add(other)
        found.append(
            Confusable(
                name=record.name,
                composition=record.composition,
                manufacturer=record.manufacturer,
                signature=record.signature,
                why=why,
                edit_distance=signals["edit_distance"],
                damerau_similarity=signals["damerau_similarity"],
                jaro_winkler=signals["jaro_winkler"],
                phonetic_match=signals["phonetic_match"],
                # Rank by the strongest single signal rather than an average:
                # a perfect phonetic match with mediocre spelling similarity is
                # exactly as dangerous as the reverse, and averaging would bury
                # both beneath a pair that is merely middling on everything.
                similarity=max(signals["damerau_similarity"], signals["jaro_winkler"]),
                source={
                    "dataset": "a_z_medicines_india",
                    "record_id": str(record.row),
                    "url": None,
                },
            )
        )

    found.sort(key=lambda c: c.similarity, reverse=True)
    found = found[:limit]

    if not found:
        message = (
            f"No product in the catalogue has a name confusable with '{name}' "
            "while containing different active ingredients."
        )
        caution = ""
    else:
        plural = "product" if len(found) == 1 else "products"
        message = (
            f"{len(found)} {plural} have names similar to '{name}' but different "
            "active ingredients."
        )
        caution = CAUTION

    return LasaResult(
        query=name, query_root=root, confusable=found, message=message, caution=caution
    )
