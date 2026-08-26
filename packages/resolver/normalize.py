"""
Normalisation of brand names, compositions, pack sizes and prices.

Everything downstream depends on this module getting one thing right: two
products with the same active ingredients at the same strengths in the same
dosage form must produce the **same composition signature**, and two products
that differ in any of those must not. That signature is what makes generic
substitution a set lookup instead of a guess, and it is the difference between
recommending a real cheaper equivalent and inventing one.

The three source datasets encode composition three different ways:

  A_Z brands      two columns of `Amoxycillin  (500mg)` / ` Clavulanic Acid (125mg)`
  Jan Aushadhi    one inline string, `Aceclofenac 100mg and Paracetamol 325mg Tablets`
  master joins    a stringified Python list of dicts

Each needs its own parser, and all three must converge on one representation.

Pack sizes matter as much as compositions, and for the same reason. A branded
strip at Rs 223 and a Jan Aushadhi pack at Rs 10 are not comparable numbers
until both are reduced to a price per tablet — comparing the printed MRPs
directly is how a system ends up reporting a 95% saving that does not exist
because one pack holds ten tablets and the other holds one.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

# --- ingredient synonyms --------------------------------------------------
# Indian labelling uses British, American and INN spellings interchangeably,
# often within one manufacturer's range. Without folding them, the same molecule
# produces different signatures and equivalent products never match.
INGREDIENT_SYNONYMS: dict[str, str] = {
    "amoxicillin": "amoxycillin",
    "acetaminophen": "paracetamol",
    "salbutamol": "albuterol",
    "cetirizine hydrochloride": "cetirizine",
    "cetirizine hcl": "cetirizine",
    "cetirizine dihydrochloride": "cetirizine",
    "diclofenac sodium": "diclofenac",
    "diclofenac potassium": "diclofenac",
    "diclofenac diethylamine": "diclofenac",
    "metformin hydrochloride": "metformin",
    "metformin hcl": "metformin",
    "ranitidine hydrochloride": "ranitidine",
    "ondansetron hydrochloride": "ondansetron",
    "chlorpheniramine maleate": "chlorpheniramine",
    "pheniramine maleate": "pheniramine",
    "azithromycin dihydrate": "azithromycin",
    "amlodipine besylate": "amlodipine",
    "amlodipine besilate": "amlodipine",
    "atorvastatin calcium": "atorvastatin",
    "pantoprazole sodium": "pantoprazole",
    "esomeprazole magnesium": "esomeprazole",
    "levocetirizine dihydrochloride": "levocetirizine",
    "montelukast sodium": "montelukast",
    "sildenafil citrate": "sildenafil",
    "tramadol hydrochloride": "tramadol",
    "ciprofloxacin hydrochloride": "ciprofloxacin",
    "norfloxacin": "norfloxacin",
    "vitamin c": "ascorbic acid",
    "vitamin b12": "cyanocobalamin",
    "vitamin d3": "cholecalciferol",
}

# Salt/ester suffixes to strip when no explicit synonym applies. The salt form
# changes the counter-ion, not the therapeutic moiety, so `Amlodipine Besylate
# 5mg` and `Amlodipine 5mg` are the same product for substitution purposes.
_SALT_SUFFIXES = (
    "hydrochloride", "hcl", "sodium", "potassium", "calcium", "magnesium",
    "sulphate", "sulfate", "phosphate", "maleate", "besylate", "besilate",
    "citrate", "tartrate", "succinate", "fumarate", "acetate", "mesylate",
    "dihydrate", "monohydrate", "trihydrate", "anhydrous", "diethylamine",
    "dihydrochloride", "orotate", "gluconate", "lactate", "nitrate", "bromide",
)

DOSAGE_FORMS = (
    "tablet", "capsule", "syrup", "suspension", "injection", "cream", "gel",
    "ointment", "drops", "solution", "powder", "sachet", "inhaler", "spray",
    "lotion", "patch", "suppository", "granules", "infusion", "eye drops",
)

# Unit conversion to a base unit, so 0.5 g and 500 mg compare equal.
_MASS_TO_MG = {"mcg": 0.001, "ug": 0.001, "mg": 1.0, "g": 1000.0, "gm": 1000.0}
_VOLUME_TO_ML = {"ml": 1.0, "l": 1000.0}

# Formulation modifiers. These describe how a tablet is made, not what is in
# it, so they must not become part of the ingredient name — otherwise
# `diclofenac gastro-resistant` and `diclofenac` are different molecules as far
# as the signature is concerned, and never match.
_MODIFIERS = (
    "gastro-resistant", "gastro resistant", "enteric coated", "enteric-coated",
    "prolonged release", "sustained release", "extended release", "delayed release",
    "modified release", "controlled release", "immediate release",
    "film coated", "film-coated", "sugar coated", "sugar-coated",
    "dispersible", "chewable", "effervescent", "orally disintegrating",
    "for oral suspension", "lyophilized", "sterile",
)
_MODIFIER_RE = re.compile(r"\b(" + "|".join(re.escape(m) for m in _MODIFIERS) + r")\b",
                          re.IGNORECASE)

_BRACKETED_RE = re.compile(r"^\s*(?P<name>[^(]+?)\s*\(\s*(?P<strength>[^)]+?)\s*\)\s*$")
_STRENGTH_RE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mcg|ug|mg|gm|g|ml|l|iu|%\s*w/w|%\s*w/v|%)",
    re.IGNORECASE,
)
_CONCENTRATION_RE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mcg|mg|g)\s*(?:/|\s+per\s+)\s*(?P<per>\d+(?:\.\d+)?)?\s*(?P<per_unit>ml|l)\b",
    re.IGNORECASE,
)

_PACK_COUNT_RE = re.compile(
    r"(?:strip|packet|pack|box|bottle|jar|tube|vial|ampoule|sachet)\s+of\s+"
    r"(?P<count>\d+(?:\.\d+)?)\s*(?P<unit>ml|gm|g|mg)?",
    re.IGNORECASE,
)
_APOSTROPHE_S_RE = re.compile(r"^\s*(?P<count>\d+)\s*'?s\s*$", re.IGNORECASE)
_BARE_QUANTITY_RE = re.compile(r"^\s*(?P<count>\d+(?:\.\d+)?)\s*(?P<unit>ml|g|gm|mg)\s*$", re.I)


@dataclass(frozen=True)
class Ingredient:
    """One active ingredient with its strength, normalised to a base unit."""

    name: str
    strength: float | None
    unit: str | None
    per_volume_ml: float | None = None
    """Set for concentrations like `30mg/5ml`, where strength alone is
    meaningless — 30mg/5ml and 30mg/15ml are different products."""

    def key(self) -> tuple:
        """Hashable identity used to build a composition signature."""
        return (self.name, self.strength, self.unit, self.per_volume_ml)

    def __str__(self) -> str:
        if self.strength is None:
            return self.name
        base = f"{self.name} {self.strength:g}{self.unit or ''}"
        return f"{base}/{self.per_volume_ml:g}ml" if self.per_volume_ml else base


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation to spaces, collapse whitespace."""
    if not text:
        return ""
    lowered = text.lower().strip()
    cleaned = re.sub(r"[^a-z0-9\s%./+-]", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def canonical_ingredient(name: str) -> str:
    """Fold an ingredient name to a canonical form.

    Applies the explicit synonym table first, then strips trailing salt or
    hydrate qualifiers. Order matters: the table holds the cases where stripping
    alone would give the wrong answer.
    """
    cleaned = normalize_text(name)
    cleaned = re.sub(r"\b(ip|bp|usp|ep)\b", "", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)

    if cleaned in INGREDIENT_SYNONYMS:
        return INGREDIENT_SYNONYMS[cleaned]

    words = cleaned.split()
    while len(words) > 1 and words[-1] in _SALT_SUFFIXES:
        words.pop()
    stripped = " ".join(words)

    return INGREDIENT_SYNONYMS.get(stripped, stripped)


def _normalise_strength(value: float, unit: str) -> tuple[float | None, str | None]:
    """Convert a strength to a base unit (mg for mass, ml for volume)."""
    unit = unit.lower().replace(" ", "")
    if unit in _MASS_TO_MG:
        return round(value * _MASS_TO_MG[unit], 6), "mg"
    if unit in _VOLUME_TO_ML:
        return round(value * _VOLUME_TO_ML[unit], 6), "ml"
    if unit.startswith("%"):
        return value, "%"
    if unit == "iu":
        return value, "iu"
    return value, unit


def parse_strength(text: str) -> tuple[float | None, str | None, float | None]:
    """Parse a strength string. Returns (value, unit, per_volume_ml).

    Concentrations are matched first: `30mg/5ml` must not be read as a bare
    30 mg, because a syrup's strength is only meaningful per volume, and
    treating the two as equal would equate products that are not
    interchangeable.
    """
    if not text:
        return None, None, None

    concentration = _CONCENTRATION_RE.search(text)
    if concentration:
        value, unit = _normalise_strength(
            float(concentration.group("value")), concentration.group("unit")
        )
        per = float(concentration.group("per") or 1.0)
        per_ml, _ = _normalise_strength(per, concentration.group("per_unit"))
        return value, unit, per_ml

    match = _STRENGTH_RE.search(text)
    if match:
        value, unit = _normalise_strength(float(match.group("value")), match.group("unit"))
        return value, unit, None

    return None, None, None


def parse_bracketed_composition(*columns: str) -> list[Ingredient]:
    """Parse the A_Z dataset format: `Amoxycillin  (500mg)`.

    Accepts several columns (the dataset splits composition across
    `short_composition1` and `short_composition2`) and handles the comma-joined
    variant found in some rows.
    """
    ingredients: list[Ingredient] = []

    for column in columns:
        if not column or not column.strip():
            continue

        # Some rows pack both ingredients into one column, comma-separated.
        # Splitting only on commas that follow a closing bracket avoids
        # breaking names that legitimately contain a comma.
        for part in re.split(r"\)\s*,", column):
            part = part.strip()
            if not part:
                continue
            if "(" in part and not part.endswith(")"):
                part += ")"

            match = _BRACKETED_RE.match(part)
            if match:
                name = canonical_ingredient(match.group("name"))
                value, unit, per_ml = parse_strength(match.group("strength"))
            else:
                name = canonical_ingredient(part)
                value, unit, per_ml = parse_strength(part)

            if name:
                ingredients.append(Ingredient(name, value, unit, per_ml))

    return ingredients


def parse_inline_composition(text: str) -> list[Ingredient]:
    """Parse the Jan Aushadhi format.

    Two naming conventions appear in the catalogue and both must work:

        Aceclofenac 100mg and Paracetamol 325mg Tablets   strength before form
        Aceclofenac Tablets IP 100 mg                     strength AFTER form

    The second is the majority case. An earlier version removed the dosage form
    *and everything after it*, which silently discarded the strength for **64.7%
    of the 2,439-row catalogue** — the resulting signature `(aceclofenac, None)`
    could never match `(aceclofenac, 100mg)` from the brand side, so generic
    substitution would have failed for two-thirds of the cheapest alternatives
    while appearing to work. Only the form *word* is removed now.

    Release and coating modifiers (`Gastro-resistant`, `Prolonged Release`) are
    stripped from the ingredient name too. They describe the formulation, not
    the molecule, and leaving them in makes `diclofenac gastro-resistant` a
    different ingredient from `diclofenac`.
    """
    if not text:
        return []

    working = _prefer_informative_parenthetical(text.strip())

    working = re.sub(r"\b(i\.?p\.?|b\.?p\.?|u\.?s\.?p\.?|e\.?p\.?|n\.?f\.?i\.?)\b", " ",
                     working, flags=re.IGNORECASE)
    working = _MODIFIER_RE.sub(" ", working)
    working = re.sub(r"\b(" + "|".join(DOSAGE_FORMS) + r")s?\b", " ", working,
                     flags=re.IGNORECASE)
    working = re.sub(r"\s+", " ", working).strip()

    if not working:
        return []

    # Handle the "names first, strengths last" layout explicitly, before any
    # fragment splitting.
    #
    # `Amoxycillin and Potassium Clavulanate Tablets IP 500mg + 125mg` has no
    # delimiter between the final name and the first strength, so a naive split
    # yields ["Amoxycillin", "Potassium Clavulanate 500mg", "125mg"] — and
    # pairing the leftovers then assigns amoxycillin 125mg and clavulanate
    # 500mg. Both doses land on the wrong drug while looking entirely plausible.
    # Peeling the trailing strength run off first keeps the two lists aligned.
    head, trailing = _split_trailing_strengths(working)
    if trailing:
        names = [
            canonical_ingredient(f)
            for f in re.split(r"\s+and\s+|\s*\+\s*|\s*,\s*|\s*&\s*", head, flags=re.IGNORECASE)
            if f.strip()
        ]
        names = [n for n in names if n]
        if len(names) == len(trailing):
            return [
                Ingredient(name, value, unit, per_ml)
                for name, (value, unit, per_ml) in zip(names, trailing)
            ]

        # Counts disagree, so the correspondence is unknowable. Return the names
        # with no strengths rather than falling through to per-fragment parsing.
        #
        # Falling through is not neutral: for `Alpha and Beta and Gamma Tablets
        # 10mg + 20mg` the split yields ["Alpha", "Beta", "Gamma 10mg", "20mg"],
        # so Gamma silently acquires 10mg purely because it happened to sit
        # adjacent to it. Dropping all the strengths loses information; keeping
        # one that is attached to the wrong drug is a fabricated dose.
        if names:
            return [Ingredient(name, None, None, None) for name in names]

    fragments = [
        f.strip()
        for f in re.split(r"\s+and\s+|\s*\+\s*|\s*,\s*|\s*&\s*", working, flags=re.IGNORECASE)
        if f.strip()
    ]

    ingredients: list[Ingredient] = []
    for fragment in fragments:
        value, unit, per_ml = parse_strength(fragment)
        name = canonical_ingredient(
            _STRENGTH_RE.sub(" ", _CONCENTRATION_RE.sub(" ", fragment))
        )
        if name:
            # A strengthless ingredient is recorded as such. That is honest —
            # the source really does not state one — and the price engine
            # reports "not comparable" rather than inventing a dose.
            ingredients.append(Ingredient(name, value, unit, per_ml))

    return ingredients


_TRAILING_STRENGTHS_RE = re.compile(
    r"(?P<head>.*?)\s*(?P<tail>"
    r"\d+(?:\.\d+)?\s*(?:mcg|ug|mg|gm|g|ml|iu)"
    r"(?:\s*(?:\+|,|and|&)\s*\d+(?:\.\d+)?\s*(?:mcg|ug|mg|gm|g|ml|iu))+"
    r")\s*$",
    re.IGNORECASE,
)


def _split_trailing_strengths(text: str) -> tuple[str, list[tuple]]:
    """Split `Amoxycillin and Clavulanate 500mg + 125mg` into head and strengths.

    Only fires when the string ends in **two or more** strengths joined by
    `+`/`,`/`and`. A single trailing strength is ambiguous — it could belong to
    the last ingredient or to all of them — and is left to normal fragment
    parsing.
    """
    match = _TRAILING_STRENGTHS_RE.match(text)
    if not match:
        return text, []

    head = match.group("head").strip()
    if not head:
        return text, []

    strengths: list[tuple] = []
    for piece in re.split(r"\s*(?:\+|,|and|&)\s*", match.group("tail"), flags=re.IGNORECASE):
        value, unit, per_ml = parse_strength(piece)
        if value is None:
            return text, []
        strengths.append((value, unit, per_ml))

    return head, strengths


def _prefer_informative_parenthetical(text: str) -> str:
    """Choose between a string's parenthetical and its surrounding text.

    Parentheses serve two opposite purposes in this catalogue:

        Diclofenac Gel IP 1.16%w/w (Diclofenac Diethylamine)
            -> the bracket names the salt; the strength is outside. Drop it.

        Co-trimoxazole (Sulphamethoxazole 200mg and Trimethoprim 40mg)
            -> the bracket IS the composition; outside is a collective name
               with no strength at all. Keep it, drop the outside.

    Deciding by which side carries strengths gets both right. Dropping
    parentheticals unconditionally — the obvious simplification — silently
    reduced every co-trimoxazole product to a strengthless `co-trimoxazole`,
    which can never match a branded equivalent.
    """
    inner_matches = re.findall(r"\(([^)]*)\)", text)
    if not inner_matches:
        return text

    outside = re.sub(r"\([^)]*\)", " ", text)
    outside_has_strength = bool(_STRENGTH_RE.search(outside))

    informative = [m for m in inner_matches if _STRENGTH_RE.search(m)]
    if informative and not outside_has_strength:
        return " and ".join(informative)

    return outside


def parse_python_literal_composition(text: str) -> list[Ingredient]:
    """Parse the master dataset's stringified list of dicts.

    `master_medicines_final.csv` stores composition as the repr of a Python
    list, e.g. `[{'name': 'aceclofenac', 'value': 100.0, 'unit': 'mg'}]`.
    `literal_eval` is used rather than `eval` — the file is data, and data must
    never be executed.
    """
    if not text or not text.strip().startswith("["):
        return []

    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []

    ingredients: list[Ingredient] = []
    for item in parsed if isinstance(parsed, list) else []:
        if not isinstance(item, dict):
            continue
        name = canonical_ingredient(str(item.get("name", "")))
        if not name:
            continue
        raw_value, raw_unit = item.get("value"), item.get("unit")
        if raw_value is not None and raw_unit:
            value, unit = _normalise_strength(float(raw_value), str(raw_unit))
        else:
            value, unit = None, None
        ingredients.append(Ingredient(name, value, unit, None))

    return ingredients


def composition_signature(ingredients: list[Ingredient]) -> tuple:
    """Canonical, order-independent identity for a composition.

    Sorted so that `Amoxycillin + Clavulanic Acid` and `Clavulanic Acid +
    Amoxycillin` are recognised as the same product — combination products are
    listed in whichever order the manufacturer chose, and equality must not
    depend on that.

    Two products sharing a signature are therapeutically interchangeable *on
    composition alone*. Dosage form is deliberately not folded in here; it is
    checked separately by the alternatives engine, because a signature match
    with a form mismatch is a real and different finding — the same molecule as
    a syrup rather than a tablet — and collapsing them would hide it.
    """
    return tuple(sorted(i.key() for i in ingredients))


def signature_string(ingredients: list[Ingredient]) -> str:
    """Human-readable signature, for display and as a dict key."""
    return " + ".join(str(i) for i in sorted(ingredients, key=lambda x: x.name))


def parse_dosage_form(text: str) -> str | None:
    """Identify the dosage form mentioned in a label string.

    Longest match wins, so `eye drops` is not truncated to `drops` — they are
    different routes of administration and not substitutable.
    """
    if not text:
        return None
    lowered = text.lower()
    for form in sorted(DOSAGE_FORMS, key=len, reverse=True):
        if form in lowered:
            return form
    return None


@dataclass(frozen=True)
class PackSize:
    """A parsed pack size, with the unit the count refers to."""

    count: float | None
    unit: str
    """'units' for countable dosage forms (tablets, capsules), otherwise 'ml'
    or 'g' for measured ones."""

    raw: str = ""

    @property
    def is_countable(self) -> bool:
        return self.unit == "units"


def parse_pack_size(label: str) -> PackSize:
    """Parse an A_Z `pack_size_label`, e.g. `strip of 10 tablets`.

    Returning the unit alongside the count is essential. `bottle of 100 ml
    Syrup` and `strip of 10 tablets` both yield a number, but dividing price by
    it means "price per ml" in one case and "price per tablet" in the other.
    Losing that distinction is how a syrup ends up compared against a tablet and
    reported as 10x overpriced.
    """
    if not label:
        return PackSize(None, "units", label or "")

    match = _PACK_COUNT_RE.search(label)
    if match:
        count = float(match.group("count"))
        raw_unit = (match.group("unit") or "").lower()
        if raw_unit in ("ml", "l"):
            unit = "ml"
        elif raw_unit in ("g", "gm", "mg"):
            unit = "g"
        else:
            unit = "units"
        return PackSize(count, unit, label)

    bare = _BARE_QUANTITY_RE.match(label)
    if bare:
        raw_unit = bare.group("unit").lower()
        unit = "ml" if raw_unit == "ml" else "g"
        return PackSize(float(bare.group("count")), unit, label)

    return PackSize(None, "units", label)


def parse_unit_size(text: str) -> PackSize:
    """Parse a Jan Aushadhi `Unit Size`, e.g. `10's`, `15 g`, `Vial & Wfi`.

    Non-numeric values such as `Vial & Wfi` return a count of None rather than a
    guess. A missing count propagates as "price per unit unavailable", which the
    price engine reports honestly instead of dividing by an assumed 1 and
    emitting a confident wrong number.
    """
    if not text:
        return PackSize(None, "units", text or "")

    apostrophe = _APOSTROPHE_S_RE.match(text)
    if apostrophe:
        return PackSize(float(apostrophe.group("count")), "units", text)

    bare = _BARE_QUANTITY_RE.match(text)
    if bare:
        raw_unit = bare.group("unit").lower()
        unit = "ml" if raw_unit == "ml" else "g"
        return PackSize(float(bare.group("count")), unit, text)

    return PackSize(None, "units", text)


def parse_price(text: str) -> float | None:
    """Parse a rupee amount, tolerating symbols, commas and stray text."""
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)

    cleaned = re.sub(r"[^\d.]", "", str(text))
    if not cleaned or cleaned.count(".") > 1:
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return value if value > 0 else None


def brand_root(name: str) -> str:
    """Reduce a product name to its brand root.

    `Augmentin 625 Duo Tablet` -> `augmentin`. Strengths, pack descriptors and
    dosage forms are stripped so that lexical matching compares brand to brand.
    They are not discarded from the system — `parse_strength` recovers them
    separately — but leaving them in the matching string lets `500` and `Tablet`
    contribute similarity between products that share nothing else, which is a
    large source of spurious matches across a 253k-row index.
    """
    text = normalize_text(name)
    text = _CONCENTRATION_RE.sub(" ", text)
    text = _STRENGTH_RE.sub(" ", text)
    text = re.sub(r"\b(" + "|".join(DOSAGE_FORMS) + r")s?\b", " ", text)
    text = re.sub(
        r"\b(strip|bottle|vial|tube|jar|sachet|pack|packet|box|of|the|and|with|plus|forte|"
        r"sr|er|xr|md|dt|cr|mr|od|ip|bp|usp)\b",
        " ",
        text,
    )
    text = re.sub(r"\b\d+(\.\d+)?\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()
