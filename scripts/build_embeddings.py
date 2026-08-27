#!/usr/bin/env python3
"""
Embed every unique composition signature with Titan and save the dense index.

Run after `build_index.py` whenever the index is rebuilt — the artifact records
the index version it was built against and `DenseIndex` refuses to load across
a mismatch, for the same reason the lexical index does: a stale artifact does
not fail, it silently ranks against a world that no longer exists.

Cost note: ~13k signatures at ~13 tokens each is ~170k input tokens through
amazon.titan-embed-text-v2:0 — fractions of a rupee per full rebuild.

    python scripts/build_embeddings.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from apps.api.config import get_settings  # noqa: E402
from packages.resolver.dense import (  # noqa: E402
    DENSE_ARTIFACT_VERSION,
    EMBEDDING_DIMENSIONS,
    TitanEmbedder,
    signature_text,
)
from packages.resolver.index import ARTIFACT_VERSION, get_index  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
logger = logging.getLogger("build_embeddings")


def main() -> int:
    settings = get_settings()
    index = get_index(settings.artifact_dir)

    # Union of brand and generic signatures, deduplicated, deterministic order.
    unique: dict[tuple, None] = {}
    for signature in index._signatures:  # noqa: SLF001 — build script, same package family
        if signature:
            unique.setdefault(signature, None)
    for generic in index.all_generics():
        if generic.signature:
            unique.setdefault(generic.signature, None)
    signatures = list(unique)
    texts = [signature_text(s) for s in signatures]
    logger.info("embedding %d unique signatures (of %d brand rows)", len(signatures), len(index))

    embedder = TitanEmbedder(
        region=settings.aws_region,
        model_id=settings.bedrock_embedding_model_id,
        dimensions=EMBEDDING_DIMENSIONS,
        access_key_id=settings.aws_access_key_id,
        secret_access_key=settings.aws_secret_access_key,
    )
    started = time.time()
    vectors = embedder.embed_many(texts)
    logger.info("embedded in %.1fs", time.time() - started)

    # Titan normalizes on request, but trust arithmetic over flags.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.clip(norms, 1e-8, None)

    np.save(settings.artifact_dir / "dense_vectors.npy", vectors)
    joblib.dump(
        {
            "dense_version": DENSE_ARTIFACT_VERSION,
            "index_version": ARTIFACT_VERSION,
            "model_id": settings.bedrock_embedding_model_id,
            "dimensions": EMBEDDING_DIMENSIONS,
            "signatures": signatures,
            "built_unix": time.time(),
        },
        settings.artifact_dir / "dense_meta.joblib",
    )
    size_mb = (settings.artifact_dir / "dense_vectors.npy").stat().st_size / 1e6
    logger.info("saved dense_vectors.npy (%.1f MB) + dense_meta.joblib", size_mb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
