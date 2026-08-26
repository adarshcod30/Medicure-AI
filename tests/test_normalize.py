"""
Tests for composition, pack-size and price normalisation.

These are the highest-stakes tests in the project. A composition signature that
is wrong in either direction causes a specific, real harm:

  * too permissive -> the system recommends a "cheaper equivalent" that is a
    different drug or a different strength
  * too strict     -> the genuine Jan Aushadhi equivalent is never found, and
    the affordability feature silently does nothing

Both failures are invisible at runtime. Only tests catch them.
"""

from __future__ import annotations

import pytest

from packages.resolver import normalize as N


# --- composition signature convergence ------------------------------------


def test_the_three_source_formats_converge_on_one_signature():
    """The whole design rests on this. Three datasets, three encodings, one key."""
    az = N.parse_bracketed_composition("Aceclofenac  (100mg) ", "  Paracetamol (325mg)")
    jan = N.parse_inline_composition("Aceclofenac 100mg and Paracetamol 325mg Tablets")
    master = N.parse_python_literal_composition(
        "[{'name': 'aceclofenac', 'value': 100.0, 'unit': 'mg'},"
        " {'name': 'paracetamol', 'value': 325.0, 'unit': 'mg'}]"
    )

    assert N.composition_signature(az) == N.composition_signature(jan)
    assert N.composition_signature(az) == N.composition_signature(master)


def test_signature_is_order_independent():
    """Combination products are listed in whichever order the maker chose."""
    a = N.parse_inline_composition("Aceclofenac 100mg and Paracetamol 325mg Tablets")
    b = N.parse_inline_composition("Paracetamol 325mg and Aceclofenac 100mg Tablets")
    assert N.composition_signature(a) == N.composition_signature(b)


def test_salt_forms_fold_to_the_same_molecule():
    """`Amlodipine Besylate 5mg` and `Amlodipine 5mg` are substitutable."""
    salt = N.parse_bracketed_composition("Amlodipine Besylate (5mg)")
    plain = N.parse_inline_composition("Amlodipine 5mg Tablets")
    assert N.composition_signature(salt) == N.composition_signature(plain)


def test_units_are_converted_to_a_common_base():
    """0.5 g and 500 mg are the same dose."""
    grams = N.parse_bracketed_composition("Amoxycillin (0.5g)")
    millis = N.parse_bracketed_composition("Amoxycillin (500mg)")
    assert N.composition_signature(grams) == N.composition_signature(millis)


def test_different_strengths_do_not_match():
    """The critical negative case — a substitution guard."""
    a = N.parse_bracketed_composition("Amlodipine (5mg)")
    b = N.parse_bracketed_composition("Amlodipine (10mg)")
    assert N.composition_signature(a) != N.composition_signature(b)


def test_concentration_is_not_reduced_to_a_bare_strength():
    """`30mg/5ml` must not equal `30mg`.

    A syrup's strength is meaningless without its volume: 30mg/5ml and
    30mg/15ml are three-fold different products, and collapsing either to a
    bare 30mg would make them interchangeable.
    """
    syrup = N.parse_bracketed_composition("Ambroxol (30mg/5ml)")
    tablet = N.parse_bracketed_composition("Ambroxol (30mg)")

    assert N.composition_signature(syrup) != N.composition_signature(tablet)
    assert syrup[0].per_volume_ml == 5.0


# --- the Jan Aushadhi naming conventions ----------------------------------


def test_strength_after_dosage_form_is_parsed():
    """Regression test for a bug that broke 64.7% of the Jan Aushadhi catalogue.

    An earlier parser removed the dosage form *and everything after it*. Since
    the majority convention is `<Ingredient> <Form> IP <Strength>`, that
    discarded the strength for two-thirds of rows — producing signatures that
    could never match the brand side, so generic substitution would have
    silently found nothing while appearing to work.
    """
    parsed = N.parse_inline_composition("Aceclofenac Tablets IP 100 mg")
    assert len(parsed) == 1
    assert parsed[0].name == "aceclofenac"
    assert parsed[0].strength == 100.0
    assert parsed[0].unit == "mg"


def test_trailing_strength_list_pairs_with_the_right_drug():
    """Regression test for swapped doses.

    `Amoxycillin and Potassium Clavulanate Tablets IP 500mg + 125mg` has no
    delimiter between the last name and the first strength, so a naive split
    produced ["Amoxycillin", "Potassium Clavulanate 500mg", "125mg"] and the
    leftover pairing gave amoxycillin 125mg / clavulanate 500mg. Both doses on
    the wrong drug, and entirely plausible-looking.
    """
    parsed = N.parse_inline_composition(
        "Amoxycillin and Potassium Clavulanate Tablets IP 500mg + 125mg"
    )
    doses = {i.name: i.strength for i in parsed}

    assert doses.get("amoxycillin") == 500.0
    assert doses.get("potassium clavulanate") == 125.0


def test_ambiguous_pairing_is_refused_rather_than_guessed():
    """When names and strengths cannot be aligned, no strength is invented."""
    parsed = N.parse_inline_composition("Alpha and Beta and Gamma Tablets 10mg + 20mg")
    # Three names, two strengths: no correspondence is derivable, so none is
    # asserted. Dropping the strengths loses information; attaching one to the
    # wrong drug fabricates a dose.
    assert {i.name for i in parsed} == {"alpha", "beta", "gamma"}
    assert all(i.strength is None for i in parsed)


def test_informative_parenthetical_is_kept():
    """`Co-trimoxazole (Sulphamethoxazole 800mg and Trimethoprim 160mg)`.

    Dropping parentheticals unconditionally reduced every co-trimoxazole
    product to a strengthless collective name.
    """
    parsed = N.parse_inline_composition(
        "Co-trimoxazole (Sulphamethoxazole 800mg and Trimethoprim 160mg) Tablets IP"
    )
    doses = {i.name: i.strength for i in parsed}

    assert doses.get("sulphamethoxazole") == 800.0
    assert doses.get("trimethoprim") == 160.0


def test_salt_naming_parenthetical_is_dropped():
    """The opposite case: the bracket names the salt, the strength is outside."""
    parsed = N.parse_inline_composition("Diclofenac Gel IP 1.16%w/w (Diclofenac Diethylamine)")
    assert len(parsed) == 1
    assert parsed[0].name == "diclofenac"
    assert parsed[0].strength == pytest.approx(1.16)


def test_formulation_modifiers_are_not_part_of_the_molecule():
    """`Gastro-resistant` describes the coating, not the drug."""
    modified = N.parse_inline_composition("Diclofenac Sodium Prolonged Release Tablets IP 100 mg")
    plain = N.parse_bracketed_composition("Diclofenac (100mg)")
    assert N.composition_signature(modified) == N.composition_signature(plain)


def test_per_ml_concentration_syntax():
    parsed = N.parse_inline_composition("Diclofenac Sodium Injection IP 25mg per ml")
    assert parsed[0].strength == 25.0
    assert parsed[0].per_volume_ml == 1.0


def test_genuinely_strengthless_products_stay_strengthless():
    """No dose is invented for a product that does not state one."""
    parsed = N.parse_inline_composition("Calamine Lotion IP")
    assert len(parsed) == 1
    assert parsed[0].name == "calamine"
    assert parsed[0].strength is None


# --- pack sizes -----------------------------------------------------------


@pytest.mark.parametrize(
    "label,count,unit",
    [
        ("strip of 10 tablets", 10.0, "units"),
        ("strip of 15 tablets", 15.0, "units"),
        ("bottle of 100 ml Syrup", 100.0, "ml"),
        ("tube of 15 gm Cream", 15.0, "g"),
        ("vial of 2 ml Injection", 2.0, "ml"),
        ("strip of 10 capsule sr", 10.0, "units"),
    ],
)
def test_pack_size_carries_its_unit(label, count, unit):
    """The unit is as important as the count.

    Both `bottle of 100 ml` and `strip of 10 tablets` yield a number, but
    dividing price by it means rupees-per-ml in one case and rupees-per-tablet
    in the other. Losing the distinction is how a syrup gets compared against a
    tablet and reported as ten times overpriced.
    """
    pack = N.parse_pack_size(label)
    assert pack.count == count
    assert pack.unit == unit


@pytest.mark.parametrize(
    "text,count,unit",
    [("10's", 10.0, "units"), ("1's", 1.0, "units"), ("15 g", 15.0, "g"), ("100 ml", 100.0, "ml")],
)
def test_unit_size_parsing(text, count, unit):
    pack = N.parse_unit_size(text)
    assert pack.count == count
    assert pack.unit == unit


def test_unparseable_pack_size_returns_none_not_a_guess():
    """`Vial & Wfi` has no count. Returning 1 would fabricate a per-unit price."""
    assert N.parse_unit_size("Vial & Wfi").count is None
    assert N.parse_unit_size("").count is None


# --- prices and brand roots -----------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [("223.42", 223.42), ("₹223.42", 223.42), ("1,234.50", 1234.50), ("", None),
     ("N/A", None), ("0", None), (10, 10.0)],
)
def test_price_parsing(raw, expected):
    assert N.parse_price(raw) == expected


@pytest.mark.parametrize(
    "name,root",
    [
        ("Augmentin 625 Duo Tablet", "augmentin duo"),
        ("Azithral 500 Tablet", "azithral"),
        ("Combiflam Tablet", "combiflam"),
        ("Pan 40 Tablet SR", "pan"),
    ],
)
def test_brand_root_strips_strengths_and_forms(name, root):
    """Leaving `500` and `Tablet` in the matching string lets unrelated products
    share similarity across a 253k-row index."""
    assert N.brand_root(name) == root


def test_python_literal_composition_does_not_execute_code():
    """The master CSV stores a Python repr. It is data, and must never run."""
    assert N.parse_python_literal_composition("[__import__('os').system('echo pwned')]") == []
    assert N.parse_python_literal_composition("not a list") == []
