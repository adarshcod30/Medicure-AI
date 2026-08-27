"""
Tests for price verification and the alternatives engine.

These guard the two claims most likely to cause real harm if wrong: what a
medicine should cost, and what may be substituted for it.
"""

from __future__ import annotations

import pytest

from packages.pharmacology.alternatives import (
    IMPLAUSIBLE_SAVING_PERCENT,
    annual_saving,
    find_alternatives,
)
from packages.pharmacology.price import CeilingPrice, CeilingPriceTable, check_price
from packages.resolver.index import DEFAULT_ARTIFACT_DIR, get_index

INDEX_AVAILABLE = (DEFAULT_ARTIFACT_DIR / "index_meta.joblib").exists()
requires_index = pytest.mark.skipif(
    not INDEX_AVAILABLE, reason="index artifacts absent; run scripts/build_index.py"
)

SIG = (("paracetamol", 500.0, "mg", None),)


class FakeTable(CeilingPriceTable):
    """A ceiling table with hand-set contents, for arithmetic tests."""

    def __init__(self, ceilings: dict):
        self._by_signature = ceilings
        self._coverage = {"rows": 1, "with_ceiling": len(ceilings), "with_notification": 0}


def ceiling(value: float, unit: str = "units", notification: str | None = None) -> CeilingPrice:
    return CeilingPrice(SIG, value, unit, source_row=0, notification=notification)


# --- price arithmetic -----------------------------------------------------


def test_overcharge_is_computed_per_unit_not_per_pack():
    """The core arithmetic. A pack price must be divided before comparing."""
    check = check_price(
        signature=SIG, market_price=22.62, pack_count=20, pack_unit="units",
        table=FakeTable({SIG: ceiling(1.01)}),
    )

    assert check.status == "verified_over_ceiling"
    assert check.market_price_per_unit == pytest.approx(1.131, abs=1e-3)
    assert check.overcharge_per_unit == pytest.approx(0.121, abs=1e-3)
    assert check.overcharge_percent == pytest.approx(11.98, abs=0.1)
    assert check.overcharge_per_pack == pytest.approx(2.42, abs=0.01)


def test_within_ceiling_is_reported_as_such():
    check = check_price(
        signature=SIG, market_price=15.0, pack_count=20, pack_unit="units",
        table=FakeTable({SIG: ceiling(1.01)}),
    )
    assert check.status == "verified_within_ceiling"
    assert check.overcharge_per_unit < 0


def test_workings_are_returned_so_the_number_can_be_checked():
    """A price claim the user cannot verify is just an assertion."""
    check = check_price(
        signature=SIG, market_price=22.62, pack_count=20, pack_unit="units",
        table=FakeTable({SIG: ceiling(1.01)}),
    )
    assert len(check.workings) >= 3
    assert any("22.62" in w and "20" in w for w in check.workings)


def test_missing_ceiling_is_stated_not_silently_skipped():
    """Most medicines are not price-controlled. That is a real answer.

    'No ceiling on record' and 'priced fairly' are different statements and must
    not be collapsed.
    """
    check = check_price(
        signature=SIG, market_price=22.62, pack_count=20, pack_unit="units",
        table=FakeTable({}),
    )
    assert check.status == "no_ceiling_on_record"
    assert check.overcharge_per_unit is None
    assert "not under price control" in check.message


def test_cross_unit_comparison_is_refused():
    """A per-ml ceiling against a per-tablet price yields nonsense."""
    check = check_price(
        signature=SIG, market_price=118.0, pack_count=100, pack_unit="ml",
        table=FakeTable({SIG: ceiling(1.01, unit="units")}),
    )
    assert check.status == "not_comparable"
    assert check.overcharge_per_unit is None


def test_implausible_ratio_is_refused_rather_than_reported():
    """A 50x ratio is a pack-size mismatch, not a pharmacy charging 50x."""
    check = check_price(
        signature=SIG, market_price=500.0, pack_count=1, pack_unit="units",
        table=FakeTable({SIG: ceiling(1.01)}),
    )
    assert check.status == "not_comparable"


def test_unreadable_pack_size_blocks_the_comparison():
    check = check_price(
        signature=SIG, market_price=22.62, pack_count=None, pack_unit="units",
        table=FakeTable({SIG: ceiling(1.01)}),
    )
    assert check.status == "insufficient_data"
    assert check.market_price_per_unit is None


def test_missing_provenance_is_disclosed_in_the_message():
    """The source CSV records no gazette notification. Say so."""
    check = check_price(
        signature=SIG, market_price=22.62, pack_count=20, pack_unit="units",
        table=FakeTable({SIG: ceiling(1.01, notification=None)}),
    )
    assert check.ceiling is not None
    assert check.ceiling.provenance_complete is False
    assert "gazette" in check.message.lower()


def test_present_provenance_is_not_flagged():
    check = check_price(
        signature=SIG, market_price=22.62, pack_count=20, pack_unit="units",
        table=FakeTable({SIG: ceiling(1.01, notification="S.O. 1575(E)")}),
    )
    assert check.ceiling.provenance_complete is True
    assert "gazette" not in check.message.lower()


# --- alternatives ---------------------------------------------------------


@requires_index
def test_alternatives_are_real_products_with_the_same_composition():
    index = get_index()
    reference = index.search("Crocin Advance Tablet", top_k=1)[0]

    result = find_alternatives(
        index,
        signature=reference.signature,
        reference_price_per_unit=reference.price_per_unit,
        reference_unit=reference.pack_unit,
        reference_form=reference.dosage_form,
        reference_row=reference.row,
    )

    assert result.alternatives
    for alternative in result.alternatives:
        assert alternative.price_per_unit is not None
        assert alternative.price_per_unit < reference.price_per_unit
        assert alternative.source["dataset"] in {"a_z_medicines_india", "jan_aushadhi_pmbjp"}


@requires_index
def test_empty_result_when_nothing_cheaper_exists():
    """The behaviour the replaced system's prompt forbade.

    "NEVER leave the cheaper_alternatives list empty" guarantees invention
    whenever nothing exists — which, for Jan Aushadhi equivalents, is ~88.5% of
    compositions.
    """
    index = get_index()
    reference = index.search("Crocin Advance Tablet", top_k=1)[0]

    # A reference price of essentially zero means nothing can undercut it.
    result = find_alternatives(
        index,
        signature=reference.signature,
        reference_price_per_unit=0.001,
        reference_unit=reference.pack_unit,
    )

    assert result.alternatives == []
    assert result.message
    assert "no" in result.message.lower() or "none" in result.message.lower()


@requires_index
def test_unknown_composition_yields_no_alternatives():
    result = find_alternatives(
        get_index(), signature=(), reference_price_per_unit=10.0
    )
    assert result.alternatives == []
    assert "composition could not be determined" in result.message


@requires_index
def test_cross_unit_alternatives_are_excluded():
    """A syrup is not an alternative to a tablet on price per unit."""
    index = get_index()
    syrup = index.search("Ascoril LS Syrup", top_k=1)[0]
    assert syrup.pack_unit == "ml"

    result = find_alternatives(
        index,
        signature=syrup.signature,
        reference_price_per_unit=syrup.price_per_unit,
        reference_unit="ml",
        reference_row=syrup.row,
    )
    assert all(a.pack_unit == "ml" for a in result.alternatives)


@requires_index
def test_implausible_savings_do_not_lead_the_list():
    """Regression test for a data-quality failure.

    `Apcil Tablet` appeared to offer amoxycillin 500 + clavulanate 125 at
    Rs 0.70 per tablet — a 97% saving — from a pack-size field inconsistent
    with its recorded price. Leading with it would send a patient to a pharmacy
    expecting a price that does not exist.
    """
    index = get_index()
    reference = index.search("Augmentin 625 Duo Tablet", top_k=1)[0]

    result = find_alternatives(
        index,
        signature=reference.signature,
        reference_price_per_unit=reference.price_per_unit,
        reference_unit=reference.pack_unit,
        reference_row=reference.row,
    )

    assert result.alternatives
    assert not result.alternatives[0].implausible
    assert result.alternatives[0].saving_percent <= IMPLAUSIBLE_SAVING_PERCENT


def test_annual_saving_states_its_assumption():
    """A projection built on an assumed dose must carry the assumption."""
    projection = annual_saving(10.33, 1.21, units_per_day=2.0)
    assert projection["assumed_units_per_day"] == 2.0
    assert projection["saving_per_year"] == pytest.approx(6657.6, abs=1.0)
    assert "may differ" in projection["caveat"]


@requires_index
def test_real_ceiling_table_reports_its_own_coverage():
    """Coverage is surfaced, not inferred from a run of empty answers."""
    from packages.pharmacology.price import get_ceiling_table

    coverage = get_ceiling_table().coverage
    assert coverage["rows"] > 8000
    assert 0.0 < coverage["ceiling_coverage"] < 1.0
    # The known gap: no notification is recorded for any row.
    assert coverage["notification_coverage"] == 0.0


@requires_index
def test_a_jan_aushadhi_product_is_told_it_is_already_the_cheap_one():
    """"Nothing is cheaper" is the same sentence for opposite situations.

    Someone holding an overpriced brand nobody undercuts and someone holding
    the government's own low-cost generic both get "no cheaper alternative
    exists" — and those call for opposite reactions. A Jan Aushadhi strip is
    the answer the affordability feature exists to point people toward.

    Case from photo_07: an unbranded Jan Aushadhi strip, catalogue code 603,
    Rs 1.75/tablet, with zero branded products sharing its composition.
    """
    from packages.resolver.normalize import composition_signature, parse_inline_composition

    index = get_index()
    signature = composition_signature(
        parse_inline_composition(
            "Cetirizine Dihydrochloride 5mg, Phenylephrine Hydrochloride 10mg "
            "and Paracetamol 325mg Tablets"
        )
    )

    result = find_alternatives(
        index, signature=signature, reference_price_per_unit=1.754, reference_unit="units"
    )

    assert result.already_generic is True
    assert result.alternatives == []
    assert "jan aushadhi" in result.message.lower()
    assert "nothing cheaper" in result.message.lower()


@requires_index
def test_a_branded_product_is_not_flagged_as_already_generic():
    """The negative control — the flag must not fire on an expensive brand."""
    index = get_index()
    reference = index.search("Crocin Advance Tablet", top_k=1)[0]

    result = find_alternatives(
        index,
        signature=reference.signature,
        reference_price_per_unit=reference.price_per_unit,
        reference_unit=reference.pack_unit,
        reference_row=reference.row,
    )

    assert result.already_generic is False
    assert result.alternatives


def test_message_pluralisation():
    """"1 products share this composition" reads as a bug to a user."""
    from packages.pharmacology.alternatives import _plural

    assert _plural(1, "product") == "1 product"
    assert _plural(3, "product") == "3 products"
