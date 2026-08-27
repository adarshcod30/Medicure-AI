"""
Cheaper equivalents by composition signature — the affordability engine.

An alternative is only offered when a product with an **identical composition
signature** exists in the dataset. Not a similar name, not a similar molecule,
not the model's recollection of what is usually substitutable: the same active
ingredients at the same strengths, present as a real row with a real price.

This is the direct answer to the failure mode in the system being replaced,
whose prompt read:

    ALWAYS give at least 2-3 cheaper alternatives.
    ALWAYS include Jan Aushadhi generic equivalent when possible.
    NEVER leave the cheaper_alternatives list empty.

An instruction never to return an empty list guarantees invention whenever no
alternative exists, and for this dataset that is most of the time: only 1,235
of 10,780 brand compositions (11.5%) have a Jan Aushadhi equivalent at all.
Under that prompt, roughly nine times in ten the model must make something up —
and a fabricated medicine name is a thing a patient may go and ask for.

So `find_alternatives` returns an empty list when there is nothing to return,
and the API says so plainly.

Dosage form is checked separately from composition. A signature match with a
form mismatch — the same molecule as a syrup rather than a tablet — is real and
useful information, but it is not a like-for-like substitution and is labelled
`form_differs` rather than being silently offered or silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from packages.resolver.index import BrandIndex, BrandRecord, GenericRecord

AlternativeKind = Literal["jan_aushadhi", "cheaper_brand"]

IMPLAUSIBLE_SAVING_PERCENT = 90.0
"""Above this, a claimed saving is more likely a data error than a bargain.

Observed in the dataset: `Apcil Tablet` appears to offer amoxycillin
500mg + clavulanic acid 125mg at Rs 0.70 per tablet, a 97% saving against
Augmentin. Real generics of that combination sell for Rs 8-12 per tablet. The
figure comes from a pack-size field that does not match the recorded price, not
from an unusually cheap manufacturer.

Such rows are flagged and ranked last rather than deleted. Deleting them would
hide a genuine data-quality problem, and a few of them are real — but leading
with one would send a patient to a pharmacy expecting a price that does not
exist."""


@dataclass
class Alternative:
    """One concrete, cheaper, compositionally identical product."""

    kind: AlternativeKind
    name: str
    price: float | None
    pack_label: str
    pack_count: float | None
    pack_unit: str
    price_per_unit: float | None
    saving_per_unit: float | None
    saving_percent: float | None
    dosage_form: str | None
    form_differs: bool
    source: dict
    manufacturer: str | None = None
    category: str | None = None
    implausible: bool = False
    """Saving so large it is more likely a source data error. See
    IMPLAUSIBLE_SAVING_PERCENT."""

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "name": self.name,
            "manufacturer": self.manufacturer,
            "price": self.price,
            "pack": {"count": self.pack_count, "unit": self.pack_unit, "label": self.pack_label},
            "price_per_unit": self.price_per_unit,
            "saving_per_unit": self.saving_per_unit,
            "saving_percent": self.saving_percent,
            "dosage_form": self.dosage_form,
            "form_differs": self.form_differs,
            "category": self.category,
            "implausible": self.implausible,
            "source": self.source,
        }


@dataclass
class AlternativesResult:
    """Everything found for one composition, plus why nothing was found."""

    reference_price_per_unit: float | None
    reference_unit: str
    alternatives: list[Alternative] = field(default_factory=list)
    jan_aushadhi_available: bool = False
    already_generic: bool = False
    """The scanned product is itself at or below the Jan Aushadhi price.

    Worth distinguishing. "No cheaper alternative exists" is the same sentence
    whether the user is holding an overpriced brand nobody undercuts or the
    government's own low-cost generic — and those call for opposite reactions.
    A Jan Aushadhi strip is the answer the affordability feature exists to point
    people toward, so telling its holder only that nothing is cheaper is
    technically true and practically useless."""
    total_same_composition: int = 0
    message: str = ""
    workings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "reference_price_per_unit": self.reference_price_per_unit,
            "reference_unit": self.reference_unit,
            "jan_aushadhi_available": self.jan_aushadhi_available,
            "already_generic": self.already_generic,
            "total_products_with_same_composition": self.total_same_composition,
            "count": len(self.alternatives),
            "alternatives": [a.to_dict() for a in self.alternatives],
            "message": self.message,
            "workings": self.workings,
        }


def _saving(reference: float | None, candidate: float | None) -> tuple[float | None, float | None]:
    if reference is None or candidate is None or reference <= 0:
        return None, None
    saving = round(reference - candidate, 4)
    return saving, round((saving / reference) * 100.0, 2)


def find_alternatives(
    index: BrandIndex,
    *,
    signature: tuple,
    reference_price_per_unit: float | None,
    reference_unit: str = "units",
    reference_form: str | None = None,
    reference_row: int | None = None,
    max_results: int = 8,
    min_saving_percent: float = 5.0,
) -> AlternativesResult:
    """Find cheaper products with an identical composition signature.

    Jan Aushadhi products are listed first when present — that is the scheme's
    entire purpose and its prices are typically far below branded equivalents —
    but they are not invented when absent.

    `min_saving_percent` filters out noise. A 2% difference is within the
    rounding error of the pack-size parsing and is not worth telling a patient
    to switch pharmacy for.
    """
    result = AlternativesResult(
        reference_price_per_unit=reference_price_per_unit,
        reference_unit=reference_unit,
    )

    if not signature:
        result.message = (
            "The composition could not be determined, so no equivalent products "
            "can be looked up."
        )
        return result

    siblings: list[BrandRecord] = index.by_signature(signature)
    generics: list[GenericRecord] = index.generics_by_signature(signature)
    result.total_same_composition = len(siblings) + len(generics)
    result.jan_aushadhi_available = bool(generics)

    # Is the scanned product already at or below the Jan Aushadhi price?
    if generics and reference_price_per_unit is not None:
        cheapest_generic = min(
            (g.price_per_unit for g in generics
             if g.price_per_unit is not None and g.pack_unit == reference_unit),
            default=None,
        )
        if cheapest_generic is not None:
            # A small tolerance: pack-size parsing and MRP rounding both
            # introduce a few percent, and flipping this flag on a 2% gap would
            # be noise rather than a finding.
            result.already_generic = reference_price_per_unit <= cheapest_generic * 1.05

    if reference_price_per_unit is None:
        result.message = (
            f"{result.total_same_composition} products share this exact composition, but "
            "this pack's per-unit price could not be worked out, so savings cannot be "
            "calculated."
        )

    found: list[Alternative] = []

    # --- Jan Aushadhi first ---
    for generic in generics:
        if generic.price_per_unit is None:
            continue
        if generic.pack_unit != reference_unit:
            continue  # per-ml against per-tablet is not a comparison

        saving, percent = _saving(reference_price_per_unit, generic.price_per_unit)
        if percent is not None and percent < min_saving_percent:
            continue

        found.append(
            Alternative(
                kind="jan_aushadhi",
                name=generic.name,
                price=generic.mrp,
                pack_label=generic.unit_size,
                pack_count=generic.pack_count,
                pack_unit=generic.pack_unit,
                price_per_unit=generic.price_per_unit,
                saving_per_unit=saving,
                saving_percent=percent,
                dosage_form=generic.dosage_form,
                form_differs=bool(
                    reference_form and generic.dosage_form and
                    generic.dosage_form != reference_form
                ),
                category=generic.category,
                implausible=bool(percent and percent > IMPLAUSIBLE_SAVING_PERCENT),
                source=generic.to_dict()["source"],
            )
        )

    # --- cheaper branded equivalents ---
    for sibling in siblings:
        if reference_row is not None and sibling.row == reference_row:
            continue
        if sibling.discontinued or sibling.price_per_unit is None:
            continue
        if sibling.pack_unit != reference_unit:
            continue

        saving, percent = _saving(reference_price_per_unit, sibling.price_per_unit)
        if percent is None or percent < min_saving_percent:
            continue

        found.append(
            Alternative(
                kind="cheaper_brand",
                name=sibling.name,
                manufacturer=sibling.manufacturer,
                price=sibling.price,
                pack_label=sibling.pack_label,
                pack_count=sibling.pack_count,
                pack_unit=sibling.pack_unit,
                price_per_unit=sibling.price_per_unit,
                saving_per_unit=saving,
                saving_percent=percent,
                dosage_form=sibling.dosage_form,
                form_differs=bool(
                    reference_form and sibling.dosage_form and
                    sibling.dosage_form != reference_form
                ),
                implausible=bool(percent and percent > IMPLAUSIBLE_SAVING_PERCENT),
                source=sibling.to_dict()["source"],
            )
        )

    # Order: plausible before suspect, Jan Aushadhi before brands, then by
    # saving. Putting `implausible` first in the key is what stops a
    # data-error row leading the list.
    found.sort(
        key=lambda a: (a.implausible, a.kind != "jan_aushadhi", -(a.saving_percent or 0.0))
    )
    result.alternatives = found[:max_results]

    if not result.message:
        result.message = _describe(result, len(generics), len(siblings))

    if reference_price_per_unit is not None:
        result.workings.append(
            f"This product: Rs {reference_price_per_unit:.4f} per unit"
        )
        for alternative in [a for a in result.alternatives if not a.implausible][:3]:
            result.workings.append(
                f"{alternative.name[:44]}: Rs {alternative.price_per_unit:.4f} per unit "
                f"-> saves Rs {alternative.saving_per_unit:.4f} "
                f"({alternative.saving_percent:.0f}%)"
            )

    return result


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    return f"{count} {singular if count == 1 else (plural or singular + 's')}"


def _describe(result: AlternativesResult, n_generics: int, n_brands: int) -> str:
    """Explain the outcome, including — especially — an empty one."""
    plausible = [a for a in result.alternatives if not a.implausible]
    if plausible:
        cheapest = plausible[0]
        lead = (
            f"Found {len(result.alternatives)} cheaper product(s) with exactly the same "
            f"composition. The cheapest is {cheapest.name}"
        )
        if cheapest.saving_percent is not None:
            lead += f", about {cheapest.saving_percent:.0f}% less per unit"
        if cheapest.kind == "jan_aushadhi":
            lead += ", available at Jan Aushadhi Kendras"
        suspect = len(result.alternatives) - len(plausible)
        if suspect:
            lead += (
                f". {suspect} further listing(s) show savings above "
                f"{IMPLAUSIBLE_SAVING_PERCENT:.0f}% and are flagged as likely data errors"
            )
        return lead + "."

    if result.already_generic:
        lead = (
            "This is already a Jan Aushadhi generic — the government's low-cost option, "
            "so there is nothing cheaper to switch to."
        )
        if n_brands == 0:
            lead += (
                " No branded product with this exact composition appears in the dataset "
                "at all."
            )
        return lead

    if n_generics == 0 and n_brands <= 1:
        return (
            "No other product with this exact composition is present in the dataset, "
            "and it has no Jan Aushadhi equivalent. There is no cheaper substitute to "
            "recommend."
        )

    if n_generics == 0:
        return (
            f"{_plural(n_brands, 'product')} share this composition, but none is "
            "meaningfully cheaper than this one. There is no Jan Aushadhi equivalent "
            "for it."
        )

    return (
        f"{_plural(n_generics + n_brands, 'product')} share this composition, but none "
        "is cheaper once pack sizes are taken into account."
    )


def annual_saving(
    price_per_unit_now: float, price_per_unit_alternative: float, *, units_per_day: float = 2.0
) -> dict:
    """Project a switch over a year of regular use.

    Per-tablet differences of a rupee or two read as trivial; the same figure
    over a year of twice-daily use is what makes an affordability decision
    concrete. The daily rate is an explicit assumption and is returned with the
    result rather than buried, because it is a guess and not a prescription.
    """
    daily = (price_per_unit_now - price_per_unit_alternative) * units_per_day
    return {
        "assumed_units_per_day": units_per_day,
        "saving_per_day": round(daily, 2),
        "saving_per_month": round(daily * 30, 2),
        "saving_per_year": round(daily * 365, 2),
        "caveat": (
            "Assumes continuous use at the stated daily amount. Your actual dose "
            "may differ — check with your doctor or pharmacist."
        ),
    }
