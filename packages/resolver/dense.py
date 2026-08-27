"""
Dense (embedding) retrieval over composition signatures.

The lexical index resolves character-level damage — `AUGMENTlN` still lands on
Augmentin because char n-grams survive a swapped glyph. What it cannot do is
semantics: "paracetamol with caffeine for headache" shares almost no n-grams
with "Paracetamol (500mg) + Caffeine (32mg)". Embeddings cover that gap.

Scope is deliberately narrow:

- Vectors are built over the **13k unique composition signatures**, not the
  253,973 brand rows. Identity here is the composition (see
  `BrandIndex.search_compositions`), and embedding every brand row would spend
  20x the storage to add only market-share noise.
- The embedding model ranks; it never asserts. Every result is a signature
  that exists in the catalogue, carrying the same provenance as a lexical hit.
  A wrong ranking is caught by the same calibrated abstention as ever.
- Fusion with the lexical ranking is **reciprocal rank fusion**, which uses
  only ordinal positions. Cosine similarities and TF-IDF scores are not
  commensurable, and this project has already measured one intuitive
  weighted-sum fusion into the bin.

Whether fused retrieval outranks lexical-only is an empirical question:
`python -m eval.bench_dense` answers it, and the default in
`apps/api/config.py` follows that measurement.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np

from .index import ARTIFACT_VERSION, DEFAULT_ARTIFACT_DIR, CompositionMatch

logger = logging.getLogger(__name__)

DENSE_ARTIFACT_VERSION = 1
EMBEDDING_DIMENSIONS = 256
"""Titan V2 supports 1024/512/256. At 13k vectors the quality difference
between 256 and 1024 is far below this benchmark's noise floor, and 256 keeps
the artifact at ~13 MB and every dot product cheap."""

RRF_K = 60
"""The standard constant from Cormack et al. — flattens the difference between
rank 1 and rank 2 enough that one system's confident mistake cannot bury the
other system's correct second place."""


def signature_text(signature: tuple) -> str:
    """Canonical text rendered from the signature tuple itself.

    Rendering from the tuple rather than a catalogue composition string means
    two brands with cosmetically different labels for the same signature embed
    to exactly the same text — the dense index inherits the resolver's
    identity rules instead of re-deriving its own.
    """
    parts = []
    for component in signature:
        ingredient = str(component[0])
        strength = component[1] if len(component) > 1 else None
        unit = component[2] if len(component) > 2 else None
        piece = ingredient
        if strength is not None:
            value = f"{strength:g}" if isinstance(strength, float) else str(strength)
            piece += f" {value}"
            if unit:
                piece += f" {unit}"
        parts.append(piece)
    return " + ".join(parts)


class TitanEmbedder:
    """amazon.titan-embed-text-v2:0 via InvokeModel.

    Titan embeds one text per call, so bulk work fans out over a thread pool;
    boto3 clients are thread-safe. Throttling gets bounded exponential backoff
    rather than an instant failure — an index build is a batch job, and a
    batch job that dies at item 9,000 of 13,000 helps nobody.
    """

    def __init__(
        self,
        *,
        region: str,
        model_id: str = "amazon.titan-embed-text-v2:0",
        dimensions: int = EMBEDDING_DIMENSIONS,
        access_key_id: str = "",
        secret_access_key: str = "",
        max_workers: int = 8,
    ) -> None:
        import boto3

        kwargs: dict = {"region_name": region}
        if access_key_id and secret_access_key:
            kwargs.update(
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
            )
        self._client = boto3.client("bedrock-runtime", **kwargs)
        self.model_id = model_id
        self.dimensions = dimensions
        self.max_workers = max_workers

    def embed_one(self, text: str, *, retries: int = 5) -> np.ndarray:
        body = json.dumps(
            {"inputText": text, "dimensions": self.dimensions, "normalize": True}
        )
        delay = 1.0
        for attempt in range(retries):
            try:
                response = self._client.invoke_model(modelId=self.model_id, body=body)
                payload = json.loads(response["body"].read())
                return np.asarray(payload["embedding"], dtype=np.float32)
            except Exception as exc:  # noqa: BLE001
                if attempt == retries - 1 or "Throttling" not in type(exc).__name__:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 16.0)
        raise RuntimeError("unreachable")

    def embed_many(self, texts: list[str], *, log_every: int = 1000) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for i, vector in enumerate(pool.map(self.embed_one, texts)):
                vectors[i] = vector
                if (i + 1) % log_every == 0:
                    logger.info("embedded %d/%d", i + 1, len(texts))
        return vectors


class DenseIndex:
    """13k signature vectors and a dot product. Deliberately not a vector DB.

    At this scale a normalized matrix multiply answers a query in well under a
    millisecond with zero infrastructure. `scripts/sync_atlas.py` can push the
    same vectors into Atlas Vector Search for a hosted deployment; the local
    matrix stays the reference implementation either way.
    """

    def __init__(self, artifact_dir: Path = DEFAULT_ARTIFACT_DIR):
        self.artifact_dir = Path(artifact_dir)
        meta_path = self.artifact_dir / "dense_meta.joblib"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"dense artifacts not found in {self.artifact_dir}. "
                "Run: python scripts/build_embeddings.py"
            )

        meta = joblib.load(meta_path)
        if meta.get("dense_version") != DENSE_ARTIFACT_VERSION:
            raise RuntimeError(
                f"dense artifact version {meta.get('dense_version')} != expected "
                f"{DENSE_ARTIFACT_VERSION}. Rebuild with: python scripts/build_embeddings.py"
            )
        if meta.get("index_version") != ARTIFACT_VERSION:
            # A stale dense index silently ranks against signatures that no
            # longer exist in the lexical index — the exact class of bug the
            # ARTIFACT_VERSION guard exists to make loud.
            raise RuntimeError(
                f"dense artifact was built against index version "
                f"{meta.get('index_version')}, current is {ARTIFACT_VERSION}. "
                "Rebuild with: python scripts/build_embeddings.py"
            )

        self.model_id: str = meta["model_id"]
        self.signatures: list[tuple] = meta["signatures"]
        self.vectors: np.ndarray = np.load(self.artifact_dir / "dense_vectors.npy")

    def __len__(self) -> int:
        return len(self.signatures)

    def search_vector(self, query_vector: np.ndarray, *, top_k: int = 25) -> list[tuple[tuple, float]]:
        """(signature, cosine) pairs, best first. Vectors are pre-normalized."""
        scores = self.vectors @ query_vector.astype(np.float32)
        count = min(top_k, scores.size)
        top = np.argpartition(-scores, count - 1)[:count]
        top = top[np.argsort(-scores[top])]
        return [(self.signatures[i], float(scores[i])) for i in top]


class DenseReranker:
    """Runtime fusion: embed the query, fuse rankings, never break the scan.

    Any failure — throttling, a network drop, missing credentials — degrades
    to the lexical ranking unchanged, with the error held on `last_error` for
    /v1/health. Dense retrieval improves ranking when it works; it must cost
    nothing when it does not.
    """

    def __init__(self, dense_index: DenseIndex, embedder: TitanEmbedder) -> None:
        self.index = dense_index
        self.embedder = embedder
        self.last_error: str | None = None

    def rerank(
        self, query: str, lexical: list[CompositionMatch], *, top_k: int = 5
    ) -> list[CompositionMatch]:
        if not lexical:
            return lexical
        try:
            query_vector = self.embedder.embed_one(query, retries=1)
            dense_hits = self.index.search_vector(query_vector, top_k=max(25, top_k))
            self.last_error = None
            return rrf_fuse(lexical, dense_hits, top_k=top_k)
        except Exception as exc:  # noqa: BLE001 — degrade, never fail the scan
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("dense rerank unavailable, lexical only: %s", exc)
            return lexical[:top_k]


def rrf_fuse(
    lexical: list[CompositionMatch],
    dense: list[tuple[tuple, float]],
    *,
    k: int = RRF_K,
    top_k: int = 5,
) -> list[CompositionMatch]:
    """Fuse the two rankings by reciprocal rank; return re-ranked lexical matches.

    Only signatures the lexical stage retrieved are returned. Dense evidence
    can *reorder* the candidate set but not *introduce* candidates the lexical
    stage never saw — a dense-only hit has no lexical score, and the
    calibrator's features are built on lexical scores. Letting unscored
    candidates through would hand the calibrator inputs it was never fitted
    on, which is how confident nonsense gets made.
    """
    if not dense:
        return lexical[:top_k]

    dense_rank = {sig: r for r, (sig, _) in enumerate(dense)}
    dense_score = dict(dense)

    def fused_score(rank_lex: int, signature: tuple) -> float:
        score = 1.0 / (k + rank_lex + 1)
        if signature in dense_rank:
            score += 1.0 / (k + dense_rank[signature] + 1)
        return score

    scored = [
        (fused_score(rank, match.signature), match) for rank, match in enumerate(lexical)
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)

    fused: list[CompositionMatch] = []
    for score, match in scored[:top_k]:
        match.dense_similarity = dense_score.get(match.signature)
        match.fused_rank_score = round(score, 6)
        fused.append(match)
    return fused
