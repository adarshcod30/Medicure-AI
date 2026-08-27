#!/usr/bin/env python3
"""
Retrieval benchmark: lexical vs dense vs reciprocal-rank fusion.

This measures the *retrieval* layer in isolation, before anything touches the
calibrated path. The decision rule for `enable_dense_retrieval` lives here:
fusion ships as a default only if it beats lexical-only on this benchmark,
because this project has twice adopted an "obvious" improvement that measured
worse (see CLAUDE.md).

Grading is exact signature equality at rank k — stricter than the
ingredient-overlap rule bench_identify uses for answers, deliberately so:
retrieval's job is to put the exactly-right composition in front of the
calibrator, and "a related composition was nearby" is not that.

Queries are the same stratified corruptions bench_identify builds, so numbers
here sit in the same world as the headline table.

    python -m eval.bench_dense --samples 300
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from apps.api.config import get_settings  # noqa: E402
from eval.bench_identify import build_queries  # noqa: E402
from packages.resolver.dense import DenseIndex, TitanEmbedder, rrf_fuse  # noqa: E402
from packages.resolver.index import get_index  # noqa: E402

POOL = 25
"""Candidate pool depth for both rankings. RRF reorders the lexical pool, so
its ceiling is lexical top-POOL accuracy — reported below so the ceiling is
visible rather than implied."""


def rank_of(signature: tuple, ranking: list[tuple]) -> int | None:
    for position, candidate in enumerate(ranking):
        if candidate == signature:
            return position
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    settings = get_settings()
    index = get_index(settings.artifact_dir)
    dense = DenseIndex(settings.artifact_dir)
    embedder = TitanEmbedder(
        region=settings.aws_region,
        model_id=dense.model_id,
        access_key_id=settings.aws_access_key_id,
        secret_access_key=settings.aws_secret_access_key,
    )

    queries = build_queries(index, args.samples, args.seed)
    print(f"{len(queries)} corrupted queries; embedding them...")
    started = time.time()
    query_vectors = embedder.embed_many([q["query"] for q in queries])
    print(f"embedded in {time.time() - started:.1f}s")

    arms = {name: defaultdict(lambda: [0, 0, 0, 0.0]) for name in ("lexical", "dense", "rrf")}
    # per severity: [top1, top5, in_pool, reciprocal_rank_sum]

    for item, query_vector in zip(queries, query_vectors):
        truth = item["truth_signature"]
        severity = item["severity"]

        lexical_matches = index.search_compositions(item["query"], top_k=POOL)
        lexical_sigs = [m.signature for m in lexical_matches]

        dense_hits = dense.search_vector(query_vector, top_k=POOL)
        dense_sigs = [s for s, _ in dense_hits]

        fused = rrf_fuse(lexical_matches, dense_hits, top_k=POOL)
        fused_sigs = [m.signature for m in fused]

        for name, sigs in (("lexical", lexical_sigs), ("dense", dense_sigs), ("rrf", fused_sigs)):
            rank = rank_of(truth, sigs)
            bucket = arms[name][severity]
            if rank is not None:
                bucket[0] += int(rank == 0)
                bucket[1] += int(rank < 5)
                bucket[2] += 1
                bucket[3] += 1.0 / (rank + 1)

    counts = defaultdict(int)
    for item in queries:
        counts[item["severity"]] += 1

    print(f"\n{'arm':8} {'severity':10} {'top-1':>7} {'top-5':>7} {'in-pool':>8} {'MRR':>6}")
    totals: dict[str, list[float]] = {}
    for name, buckets in arms.items():
        agg = [0, 0, 0, 0.0]
        for severity in ("light", "moderate", "heavy"):
            n = counts[severity]
            b = buckets[severity]
            agg = [a + x for a, x in zip(agg, b)]
            print(
                f"{name:8} {severity:10} {b[0]/n:7.1%} {b[1]/n:7.1%} {b[2]/n:8.1%} {b[3]/n:6.3f}"
            )
        n = len(queries)
        totals[name] = [agg[0] / n, agg[1] / n, agg[2] / n, agg[3] / n]
        print(
            f"{name:8} {'ALL':10} {totals[name][0]:7.1%} {totals[name][1]:7.1%} "
            f"{totals[name][2]:8.1%} {totals[name][3]:6.3f}"
        )
        print()

    out = REPO_ROOT / "eval" / "results" / "bench_dense.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "samples": len(queries),
                "seed": args.seed,
                "pool": POOL,
                "totals": {k: dict(zip(("top1", "top5", "in_pool", "mrr"), v)) for k, v in totals.items()},
            },
            indent=2,
        )
    )
    print(f"written {out}")

    verdict = "rrf" if totals["rrf"][0] > totals["lexical"][0] else "lexical"
    print(f"\ntop-1 winner: {verdict} "
          f"(lexical {totals['lexical'][0]:.1%} vs rrf {totals['rrf'][0]:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
