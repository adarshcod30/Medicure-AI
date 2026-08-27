"""
Tests for the resolver: index search, corruption modelling and calibration.

Index tests are skipped when the artifacts are absent, so a fresh clone can run
the suite before `scripts/build_index.py` has been executed.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from packages.resolver import corruption
from packages.resolver.calibrate import (
    Calibrator,
    expected_calibration_error,
    extract_features,
    risk_coverage,
)
from packages.resolver.index import DEFAULT_ARTIFACT_DIR, CompositionMatch, get_index

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

INDEX_AVAILABLE = (DEFAULT_ARTIFACT_DIR / "index_meta.joblib").exists()
requires_index = pytest.mark.skipif(
    not INDEX_AVAILABLE, reason="index artifacts absent; run scripts/build_index.py"
)


# --- corruption model -----------------------------------------------------


def test_corruption_is_severity_ordered():
    """Heavier profiles must actually damage more, or the eval means nothing."""
    rng = random.Random(0)
    name = "Augmentin 625 Duo Tablet"

    def mean_distance(profile, trials=60):
        from rapidfuzz.distance import Levenshtein

        return np.mean(
            [Levenshtein.distance(name, corruption.corrupt(name, profile, rng))
             for _ in range(trials)]
        )

    light = mean_distance(corruption.CorruptionProfile.light())
    moderate = mean_distance(corruption.CorruptionProfile.moderate())
    heavy = mean_distance(corruption.CorruptionProfile.heavy())

    assert light < moderate < heavy


def test_corruption_uses_realistic_confusions():
    """Substitutions must come from the OCR confusion table, not random noise.

    A matcher tuned against uniform noise learns robustness to errors that never
    occur while staying brittle to the ones that do.
    """
    rng = random.Random(1)
    profile = corruption.CorruptionProfile(substitution_rate=1.0, deletion_rate=0.0,
                                           insertion_rate=0.0, case_flip_rate=0.0,
                                           space_damage_rate=0.0)
    result = corruption.corrupt("mill", profile, rng)

    # Every character of "mill" has confusions; the result must be built from
    # them rather than from arbitrary letters.
    assert result != "mill"
    assert any(c in result for c in ("rn", "nn", "in", "1", "I", "i", "l", "t"))


def test_truncation_favours_the_leading_edge():
    """Strips tear at the perforation and the brand name sits at the top."""
    rng = random.Random(2)
    text = "ABCDEFGHIJKLMNOP"
    front = back = 0
    for _ in range(200):
        result = corruption._truncate(text, 0.4, rng)
        if text.endswith(result):
            front += 1
        elif text.startswith(result):
            back += 1
    assert front > back


def test_make_pairs_labels_severity():
    pairs = corruption.make_pairs(["Crocin 500"], per_profile=2, seed=3)
    assert len(pairs) == 6
    assert {p[2] for p in pairs} == {"light", "moderate", "heavy"}
    assert all(p[1] == "Crocin 500" for p in pairs)


# --- index ----------------------------------------------------------------


@requires_index
def test_index_loads_the_full_catalogue():
    index = get_index()
    stats = index.stats()
    assert stats["brands"] > 250_000
    assert stats["generics"] > 2_000


@requires_index
@pytest.mark.parametrize(
    "query,expected_ingredient",
    [
        ("Augmentin 625", "amoxycillin"),
        ("crocin 500", "paracetamol"),
        ("Pantop 40", "pantoprazole"),
        ("AUGMENTlN 625 DUO", "amoxycillin"),
    ],
)
def test_known_brands_resolve_to_the_right_molecule(query, expected_ingredient):
    matches = get_index().search_compositions(query, top_k=3)
    assert matches, f"no candidates for {query!r}"
    assert expected_ingredient in matches[0].label


@requires_index
def test_crocin_regression():
    """Regression test for a fusion bug.

    Under a weighted sum of name and composition similarity, `crocin 500`
    returned azithromycin: "500" matched dozens of `...500mg` compositions well
    enough to outweigh a decisive brand-name hit. A separate support bonus then
    compounded it by rewarding azithromycin's 39 selling brands over
    paracetamol's 2. Both are fixed — max-based fusion, and ranking by best
    match rather than by market share.
    """
    matches = get_index().search_compositions("crocin 500", top_k=3)
    assert "paracetamol" in matches[0].label
    assert "azithromycin" not in matches[0].label


@requires_index
def test_nonsense_scores_far_below_real_matches():
    """The separation calibration depends on."""
    index = get_index()
    real = index.search_compositions("Augmentin 625", top_k=1)
    nonsense = index.search_compositions("zzzznotarealmedicine", top_k=1)

    assert real
    assert real[0].top_similarity > 0.7
    if nonsense:
        assert nonsense[0].top_similarity < 0.5


@requires_index
def test_composition_search_beats_brand_row_search_on_corrupted_input():
    """The core measurement justifying composition as the output unit.

    253,973 brands share 10,780 compositions, so brand-row accuracy is capped
    near 1/24 when the brand name is unreadable. Composition is also the unit
    every downstream feature actually needs.
    """
    index = get_index()
    rng = random.Random(11)
    rows = rng.sample(range(len(index)), 60)
    profile = corruption.CorruptionProfile.moderate()

    row_hits = signature_hits = considered = 0
    for row in rows:
        record = index.record(row)
        if not record.signature:
            continue
        considered += 1
        query = (
            f"{corruption.corrupt(record.name, profile, rng)} "
            f"{corruption.corrupt(record.composition, profile, rng)}"
        )
        matches = index.search_compositions(query, top_k=1)
        if matches:
            signature_hits += matches[0].signature == record.signature
            row_hits += matches[0].best_row == row

    assert considered > 20
    assert signature_hits > row_hits


@requires_index
def test_signature_lookup_finds_substitutable_products():
    index = get_index()
    matches = index.search_compositions("Augmentin 625", top_k=1)
    signature = matches[0].signature

    siblings = index.by_signature(signature, limit=10)
    assert len(siblings) > 1
    assert all(s.signature == signature for s in siblings)


# --- calibration ----------------------------------------------------------


def _match(similarity: float, support: int = 1) -> CompositionMatch:
    return CompositionMatch(
        signature=(("x", 1.0, "mg", None),),
        label="x 1mg",
        best_row=0,
        best_name="X",
        top_similarity=similarity,
        aggregate_score=similarity,
        support=support,
    )


def test_margin_separates_equal_top_scores():
    """The feature a bare similarity threshold throws away.

    0.62 with the runner-up at 0.61 means two equally good answers; 0.62 with
    the runner-up at 0.20 is a clean match. Same similarity, opposite meaning.
    """
    ambiguous = extract_features([_match(0.62), _match(0.61)], "q")
    clean = extract_features([_match(0.62), _match(0.20)], "q")

    margin_index = 1
    assert clean[margin_index] > ambiguous[margin_index]


def test_expected_calibration_error_is_zero_for_perfect_predictions():
    probabilities = np.array([0.0, 0.0, 1.0, 1.0])
    correct = np.array([0, 0, 1, 1])
    ece, bins = expected_calibration_error(probabilities, correct, bins=10)
    assert ece == pytest.approx(0.0, abs=1e-9)
    assert bins


def test_expected_calibration_error_detects_overconfidence():
    """A model claiming 0.95 while being right half the time must score badly."""
    probabilities = np.full(100, 0.95)
    correct = np.array([1] * 50 + [0] * 50)
    ece, _ = expected_calibration_error(probabilities, correct)
    assert ece > 0.4


def test_risk_coverage_trades_coverage_for_precision():
    rng = np.random.default_rng(0)
    probabilities = rng.uniform(0, 1, 1000)
    # Correctness correlates with confidence, as a working calibrator implies.
    correct = (rng.uniform(0, 1, 1000) < probabilities).astype(int)

    curve = risk_coverage(probabilities, correct, targets=(0.8, 0.9))
    if curve["p80"]["achievable"] and curve["p90"]["achievable"]:
        assert curve["p90"]["coverage"] <= curve["p80"]["coverage"]


def test_unfitted_calibrator_does_not_present_similarity_as_probability():
    """An uncalibrated score must be flagged, never dressed up as a probability."""
    calibrator = Calibrator.unfitted()
    assert not calibrator.is_fitted
    assert calibrator.probability([_match(0.9)], "q") == pytest.approx(0.9)


def test_decide_returns_three_states():
    """'ambiguous' is a distinct message from 'abstained'.

    One means "which of these two is it?", the other "retake the photo". They
    call for different user actions.
    """
    calibrator = Calibrator.unfitted()
    calibrator.threshold = 0.8

    # A query with real lexical support, so the numeric-only gate is not what
    # is being tested here — see test_a_numeric_only_match_is_never_confident.
    query = "augmentin duo amoxycillin clavulanic tablets"

    assert calibrator.decide([_match(0.95)], query)[0] == "confident"
    assert calibrator.decide([_match(0.55)], query)[0] == "ambiguous"
    assert calibrator.decide([_match(0.10)], query)[0] == "abstained"
    assert calibrator.decide([], query)[0] == "abstained"


@pytest.mark.skipif(
    not (DEFAULT_ARTIFACT_DIR / "calibrator.joblib").exists(),
    reason="calibrator absent; run scripts/fit_calibrator.py",
)
def test_fitted_calibrator_round_trips():
    calibrator = Calibrator.load(DEFAULT_ARTIFACT_DIR / "calibrator.joblib")
    assert calibrator.is_fitted
    assert 0.0 < calibrator.threshold <= 1.0

    high = calibrator.probability([_match(0.95), _match(0.10)], "augmentin 625 duo")
    low = calibrator.probability([_match(0.30), _match(0.29)], "zz")
    assert high > low


# --- packaging boilerplate ------------------------------------------------


def test_boilerplate_filtering_keeps_ingredients_and_drops_storage_text():
    """Real photos returned the storage paragraph instead of the composition.

    'store in a cool dry place', 'keep out of reach of children' appears on
    every pack, identifies nothing, and crowded out the composition tokens.
    """
    from packages.perception import boilerplate

    vocabulary = {"belladonna", "paraffin", "atropine", "paracetamol"}
    stopwords = boilerplate.build_stopwords(vocabulary)

    tokens = ["store", "cool", "dry", "place", "children", "paracetamol", "belladonna", "mg"]
    kept = boilerplate.filter_tokens(tokens, stopwords)

    assert "paracetamol" in kept and "belladonna" in kept
    assert "store" not in kept and "children" not in kept


def test_function_words_are_filtered_even_inside_ingredient_names():
    """Corpus frequency is the wrong test for 'and' / 'from' / 'with'.

    They appear in under 2% of compositions (only in names like 'water for
    injection'), so a frequency rule marks them discriminative and protects
    them — while they appear in essentially every OCR read of a pack.
    """
    from packages.perception import boilerplate

    stopwords = boilerplate.build_stopwords({"and", "from", "with", "water", "belladonna"})
    assert "and" in stopwords and "from" in stopwords and "with" in stopwords
    # A genuine ingredient word stays protected.
    assert "water" not in stopwords


def test_dose_and_form_words_are_never_filtered():
    from packages.perception import boilerplate

    stopwords = boilerplate.build_stopwords(set())
    for keeper in ("mg", "ml", "tablet", "capsule", "syrup", "injection"):
        assert keeper not in stopwords


@requires_index
def test_discriminative_vocabulary_excludes_ubiquitous_words():
    from packages.resolver.index import get_index

    vocabulary = get_index().discriminative_vocabulary(max_document_frequency=0.02)
    assert "belladonna" in vocabulary
    assert len(vocabulary) > 500


def test_a_numeric_only_match_is_never_confident():
    """Regression test for a measured silent failure.

    On a low-resolution product shot, OCR returned
    ['x0.035mg', 'x0.04mg', 'x0.03mg', 'ree', 'ore', 'tens', 'tthe'].
    0.035mg is the exact strength of ethinyl estradiol — distinctive enough to
    produce a strong wide-margin match — and the calibrator scored it 0.575
    confident and wrong.

    The query-quality features miss it: `x0.035mg` is eight characters, so mean
    token length looks healthy. The problem is that no WORD supported the match.
    """
    calibrator = Calibrator.unfitted()
    calibrator.threshold = 0.5

    numeric_query = "x0.035mg x0.04mg x0.03mg ree ore tens tthe"
    status, _ = calibrator.decide([_match(0.95), _match(0.20)], numeric_query)
    assert status == "ambiguous", "a match resting on numerals must not be confident"

    worded_query = "combiflam ibuprofen paracetamol tablets sanofi"
    status, _ = calibrator.decide([_match(0.95), _match(0.20)], worded_query)
    assert status == "confident"


def test_lexical_support_counts_only_real_words():
    from packages.resolver.calibrate import Calibrator as C

    assert C.lexical_support("x0.035mg 500mg 10mg") == 0
    assert C.lexical_support("ree ore mg") == 0          # all under 4 chars
    assert C.lexical_support("paracetamol tablets ip") == 2
