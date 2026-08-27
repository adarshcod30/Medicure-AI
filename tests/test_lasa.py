"""
Tests for look-alike/sound-alike detection.

The scoring and exclusion rules are pure functions and are tested directly.
The behaviour that needs the catalogue — that a real confusable pair is found,
and that same-composition products are never reported — runs behind the
index-artifact skip.
"""

from __future__ import annotations

import pytest

from packages.pharmacology.lasa import (
    DAMERAU_SIMILARITY_FLOOR,
    JARO_WINKLER_FLOOR,
    MIN_ROOT_LENGTH,
    damerau_similarity,
    find_confusable,
    reasons,
    same_brand_family,
    score,
)
from packages.resolver.index import DEFAULT_ARTIFACT_DIR, get_index

INDEX_AVAILABLE = (DEFAULT_ARTIFACT_DIR / "index_meta.joblib").exists()
requires_index = pytest.mark.skipif(
    not INDEX_AVAILABLE, reason="index artifacts absent; run scripts/build_index.py"
)


# --- scoring --------------------------------------------------------------


def test_identical_strings_score_one():
    assert damerau_similarity("testium", "testium") == 1.0


def test_similarity_is_normalised_by_the_longer_string():
    # one substitution in seven characters
    assert damerau_similarity("testium", "testiun") == pytest.approx(6 / 7)


def test_empty_strings_do_not_divide_by_zero():
    assert damerau_similarity("", "") == 0.0


def test_transposition_counts_as_one_edit():
    """Damerau, not plain Levenshtein: 'tset' from 'test' is one slip."""
    assert score("test", "tset")["edit_distance"] == 1


# --- reason reporting -----------------------------------------------------


def test_far_apart_names_produce_no_reasons():
    assert reasons(score("testium", "quixophan")) == []


def test_close_spelling_is_reported_as_spelling():
    signals = score("testium", "testiun")
    assert signals["damerau_similarity"] >= DAMERAU_SIMILARITY_FLOOR
    assert "spelling" in reasons(signals)


def test_shared_prefix_is_reported_as_prefix():
    signals = score("testiumol", "testiumide")
    assert signals["jaro_winkler"] >= JARO_WINKLER_FLOOR
    assert "prefix" in reasons(signals)


# --- brand-family exclusion ----------------------------------------------


def test_identical_roots_are_the_same_family():
    assert same_brand_family("testium", "testium")


def test_a_trailing_qualifier_is_still_the_same_family():
    """The regression this guard exists for.

    'augmentin' vs 'augmentin duo' scored 0.94 and flooded out genuine
    cross-brand look-alikes. Same brand, different line extension.
    """
    assert same_brand_family("testium", "testium duo")
    assert same_brand_family("testium forte", "testium")


def test_different_brands_are_not_the_same_family():
    assert not same_brand_family("testium", "mockium")
    assert not same_brand_family("celebrex", "celexa")


# --- catalogue behaviour --------------------------------------------------


@requires_index
def test_a_known_confusable_pair_is_found():
    """Celebrex/Celexa — a documented dispensing-confusion pair, both stocked.

    Discovered from the catalogue rather than hard-coded: the test asserts the
    relationship, not a row number.
    """
    index = get_index()
    celebrex = index.search("Celebrex 200mg Capsule", top_k=1)[0]

    result = find_confusable(
        index,
        name=celebrex.name,
        signature=celebrex.signature,
        exclude_row=celebrex.row,
        limit=10,
    )

    names = " ".join(c.name.lower() for c in result.confusable)
    assert "celexa" in names
    assert result.caution


@requires_index
def test_products_sharing_a_composition_are_never_reported():
    """The exclusion that defines the feature.

    Same composition means the same medicine under another label — that is
    substitution, which `alternatives.py` handles. Confusion is only dangerous
    when the names are close and the contents differ.
    """
    index = get_index()
    reference = index.search("Augmentin 625 Duo Tablet", top_k=1)[0]

    result = find_confusable(
        index,
        name=reference.name,
        signature=reference.signature,
        exclude_row=reference.row,
        limit=25,
    )

    assert all(c.signature != reference.signature for c in result.confusable)


@requires_index
def test_every_finding_carries_provenance():
    index = get_index()
    reference = index.search("Lasix 40mg Tablet", top_k=1)[0]
    result = find_confusable(
        index, name=reference.name, signature=reference.signature, exclude_row=reference.row
    )
    for finding in result.confusable:
        assert finding.source["dataset"] == "a_z_medicines_india"
        assert finding.source["record_id"]
        assert finding.why


@requires_index
def test_an_empty_result_carries_no_caution():
    """No findings means no warning text — nothing to caution about."""
    index = get_index()
    reference = index.search("Crocin Advance Tablet", top_k=1)[0]
    result = find_confusable(
        index,
        name=reference.name,
        signature=reference.signature,
        exclude_row=reference.row,
        limit=25,
    )
    if not result.confusable:
        assert result.caution == ""
        assert "No product" in result.message


def test_a_very_short_name_is_refused_rather_than_guessed():
    """Below four characters nearly everything is one edit from everything."""
    class Stub:
        def search(self, *a, **k):  # pragma: no cover - must never be reached
            raise AssertionError("index must not be queried for a too-short root")

    result = find_confusable(Stub(), name="Ab", signature=())
    assert result.confusable == []
    assert str(MIN_ROOT_LENGTH) in result.message
