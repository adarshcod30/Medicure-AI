#!/usr/bin/env python3
"""
Push composition signatures + dense vectors into MongoDB Atlas Vector Search.

The local numpy matrix in `packages/resolver/dense.py` is the reference
implementation — 13k vectors need no infrastructure. This sync exists for the
hosted deployment story: with the catalogue in Atlas, the API host can run
without the 13 MB artifact, and the vectors are queryable from anywhere.

Idempotent: documents are keyed by a stable hash of the signature and
replaced, not appended. Run it after every `build_embeddings.py`.

Requires MONGODB_URI in .env to point at a real deployment (the localhost
default is refused for the vector index step, which is an Atlas feature).

    python scripts/sync_atlas.py
"""

from __future__ import annotations

import hashlib
import logging
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from apps.api.config import get_settings  # noqa: E402
from packages.resolver.dense import DenseIndex, signature_text  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
logger = logging.getLogger("sync_atlas")

COLLECTION = "composition_vectors"
VECTOR_INDEX = "composition-vector-index"


def signature_key(signature: tuple) -> str:
    return hashlib.sha1(repr(signature).encode("utf-8")).hexdigest()


def main() -> int:
    settings = get_settings()

    from pymongo import MongoClient, ReplaceOne

    client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    db = client[settings.mongodb_database]

    dense = DenseIndex(settings.artifact_dir)
    logger.info("syncing %d signature vectors to %s.%s", len(dense), settings.mongodb_database, COLLECTION)

    operations = []
    for signature, vector in zip(dense.signatures, dense.vectors):
        key = signature_key(signature)
        operations.append(
            ReplaceOne(
                {"_id": key},
                {
                    "_id": key,
                    "signature": [list(c) for c in signature],
                    "text": signature_text(signature),
                    "embedding": [float(x) for x in np.asarray(vector)],
                    "model_id": dense.model_id,
                    "synced_unix": time.time(),
                },
                upsert=True,
            )
        )

    collection = db[COLLECTION]
    for start in range(0, len(operations), 1000):
        result = collection.bulk_write(operations[start : start + 1000], ordered=False)
        logger.info("batch %d: upserted=%d modified=%d",
                    start // 1000, result.upserted_count, result.modified_count)

    if "localhost" in settings.mongodb_uri or "127.0.0.1" in settings.mongodb_uri:
        logger.warning(
            "MONGODB_URI is local — documents synced, but the vector search "
            "index is an Atlas feature and was not created. Point MONGODB_URI "
            "at Atlas and re-run."
        )
        return 0

    try:
        from pymongo.operations import SearchIndexModel

        existing = {i["name"] for i in collection.list_search_indexes()}
        if VECTOR_INDEX in existing:
            logger.info("vector index '%s' already exists", VECTOR_INDEX)
        else:
            collection.create_search_index(
                SearchIndexModel(
                    definition={
                        "fields": [
                            {
                                "type": "vector",
                                "path": "embedding",
                                "numDimensions": int(dense.vectors.shape[1]),
                                "similarity": "cosine",
                            }
                        ]
                    },
                    name=VECTOR_INDEX,
                    type="vectorSearch",
                )
            )
            logger.info("vector index '%s' creation requested (Atlas builds it async)", VECTOR_INDEX)
    except Exception as exc:  # noqa: BLE001 — index creation permissions vary by tier
        logger.warning(
            "could not create the vector search index programmatically (%s). "
            "Create it once in the Atlas UI: collection %s, field 'embedding', "
            "%d dimensions, cosine.",
            exc, COLLECTION, int(dense.vectors.shape[1]),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
