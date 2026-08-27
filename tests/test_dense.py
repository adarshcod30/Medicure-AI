"""
Tests for dense retrieval: rendering, fusion, and the version guard.

Everything here runs without artifacts or network — the Titan-dependent parts
(embedding quality, the fused benchmark) are measured by eval/bench_dense.py,
which is a benchmark and not a test precisely because its outcome is a
measurement, not an invariant.
"""

from __future__ import annotations

import joblib
import numpy as np
import pytest

from packages.resolver.dense import (
    DENSE_ARTIFACT_VERSION,
    DenseIndex,
    rrf_fuse,
    signature_text,
)
from packages.resolver.index import ARTIFACT_VERSION, CompositionMatch


def match(signature: tuple, rank_hint: int) -> CompositionMatch:
    return CompositionMatch(
        signature=signature,
        label=f"label-{rank_hint}",
        best_row=rank_hint,
        best_name=f"brand-{rank_hint}",
        top_similarity=1.0 - rank_hint * 0.1,
        aggregate_score=1.0 - rank_hint * 0.1,
        support=1,
    )


SIG_A = (("testium", 500.0, "mg", None),)
SIG_B = (("mockazole", 250.0, "mg", None),)
SIG_C = (("placebonol", 10.0, "mg", None),)


# --- signature_text -------------------------------------------------------


def test_signature_text_renders_strength_and_unit():
    assert signature_text(SIG_A) == "testium 500 mg"


def test_signature_text_joins_components_stably():
    combined = SIG_A + SIG_B
    assert signature_text(combined) == "testium 500 mg + mockazole 250 mg"


def test_signature_text_handles_missing_strength():
    assert signature_text((("testium", None, None, None),)) == "testium"


def test_signature_text_formats_floats_cleanly():
    """62.5 stays 62.5 and 500.0 becomes 500 — Titan should not embed '.0'."""
    assert signature_text((("testium", 62.5, "mg", None),)) == "testium 62.5 mg"


# --- reciprocal rank fusion ----------------------------------------------


def test_rrf_without_dense_evidence_is_the_lexical_ranking():
    lexical = [match(SIG_A, 0), match(SIG_B, 1)]
    fused = rrf_fuse(lexical, [], top_k=5)
    assert [m.signature for m in fused] == [SIG_A, SIG_B]
    assert fused[0].dense_similarity is None


def test_rrf_promotes_a_candidate_both_rankings_support():
    """Lexical rank 2 + dense rank 1 must beat lexical rank 1 + dense absent."""
    lexical = [match(SIG_A, 0), match(SIG_B, 1), match(SIG_C, 2)]
    dense = [(SIG_B, 0.93), (SIG_C, 0.55)]
    fused = rrf_fuse(lexical, dense, top_k=3)
    assert fused[0].signature == SIG_B
    assert fused[0].dense_similarity == pytest.approx(0.93)
    assert fused[0].fused_rank_score is not None


def test_rrf_never_introduces_a_candidate_lexical_did_not_retrieve():
    """The property the calibrator depends on: dense reorders, never adds.

    A dense-only candidate has no lexical score, and the calibrator was fitted
    on lexical scores — letting one through would hand it inputs from outside
    its training distribution.
    """
    lexical = [match(SIG_A, 0)]
    dense = [(SIG_B, 0.99), (SIG_A, 0.5)]
    fused = rrf_fuse(lexical, dense, top_k=5)
    assert [m.signature for m in fused] == [SIG_A]


def test_rrf_is_deterministic():
    lexical = [match(SIG_A, 0), match(SIG_B, 1)]
    dense = [(SIG_B, 0.9)]
    first = [m.signature for m in rrf_fuse(lexical, dense, top_k=2)]
    second = [m.signature for m in rrf_fuse(lexical, dense, top_k=2)]
    assert first == second


# --- artifact guards ------------------------------------------------------


def test_dense_index_refuses_missing_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_embeddings"):
        DenseIndex(tmp_path)


def test_dense_index_refuses_stale_index_version(tmp_path):
    """Built against index v(N-1) must refuse to load under v(N).

    The failure this prevents is silent: signatures that no longer exist in
    the lexical index would still rank, and fusion would quietly join on
    nothing.
    """
    joblib.dump(
        {
            "dense_version": DENSE_ARTIFACT_VERSION,
            "index_version": ARTIFACT_VERSION - 1,
            "model_id": "m",
            "signatures": [SIG_A],
        },
        tmp_path / "dense_meta.joblib",
    )
    np.save(tmp_path / "dense_vectors.npy", np.zeros((1, 4), dtype=np.float32))
    with pytest.raises(RuntimeError, match="index version"):
        DenseIndex(tmp_path)


def test_dense_index_refuses_wrong_dense_version(tmp_path):
    joblib.dump(
        {
            "dense_version": DENSE_ARTIFACT_VERSION + 1,
            "index_version": ARTIFACT_VERSION,
            "model_id": "m",
            "signatures": [SIG_A],
        },
        tmp_path / "dense_meta.joblib",
    )
    np.save(tmp_path / "dense_vectors.npy", np.zeros((1, 4), dtype=np.float32))
    with pytest.raises(RuntimeError, match="dense artifact version"):
        DenseIndex(tmp_path)
