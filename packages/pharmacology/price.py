"""
Price verification against NPPA ceiling prices — capability C3.

Everything here is arithmetic over retrieved records. No language model is
involved, and that is the point. Asked whether Augmentin 625 is overpriced, an
LLM produces a confident number it cannot have: DPCO ceiling prices change by
individual gazette order (the most recent revised 907 formulations effective
1 April 2026), so the correct value is not in any model's weights. The
reference implementation this project replaces went further and *instructed*
the model to invent one — `medicine_prompt.txt` says "add approximate Indian
price inside brackets".

Three rules govern this module:

1. **Compute, never recall.** Every rupee figure is derived from a retrieved
   record by explicit arithmetic, and the inputs are returned alongside the
   result so the user can check it.

2. **Per-unit or nothing.** A branded strip at Rs 223 and a Jan Aushadhi pack
   at Rs 10 are not comparable until both are per tablet. Pack sizes carry
   their unit, and comparing across units (per-ml against per-tablet) is
   refused rather than performed.

3. **Say when there is no ceiling.** Only about 17% of the products in the
   dataset have an NPPA price on record — most medicines are simply not under
   price control. "No ceiling price on record" is the correct answer for the
   rest, and it is a materially different statement from "priced fairly".

**Known data gap.** The `nppa_notif` and `nppa_date` columns of
`master_medicines_final.csv` are entirely empty, so a ceiling price taken from
it cannot be traced to the order that set it. `CeilingPrice.provenance_complete`
reports this per-record, and the API surfaces it. Closing the gap needs the
NPPA gazette scrape — the agency publishes per-order PDFs rather than a
consolidated file.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from packages.resolver.normalize import (
    composition_signature,
    parse_pack_size,
    parse_price,
    parse_python_literal_composition,
)

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

PriceStatus = Literal[
    "verified_within_ceiling",
    "verified_over_ceiling",
    "no_ceiling_on_record",
    "not_comparable",
    "insufficient_data",
]

# A market price this far above the ceiling is treated as a data error rather
# than a violation. Ratios of 50x in the source data come from pack-size
# mismatches (a per-unit ceiling compared against a whole-pack price), not from
# pharmacies charging fifty times the legal maximum. Reporting those as
# overcharging would be both wrong and alarming.
IMPLAUSIBLE_RATIO = 20.0


@dataclass
class CeilingPrice:
    """An NPPA ceiling price for one composition."""

    signature: tuple
    ceiling_per_unit: float
    unit: str
    source_row: int
    notification: str | None = None
    notified_on: str | None = None

    @property
    def provenance_complete(self) -> bool:
        """Whether this ceiling can be traced to the order that set it.

        Currently false for every record, because the source CSV's notification
        and date columns are empty. Surfaced rather than hidden: a price claim
        without a citation is weaker evidence, and the user is entitled to know
        which kind they are looking at.
        """
        return bool(self.notification)

    def to_dict(self) -> dict:
        return {
            "ceiling_per_unit": self.ceiling_per_unit,
            "unit": self.unit,
            "notification": self.notification,
            "notified_on": self.notified_on,
            "provenance_complete": self.provenance_complete,
            "source": {
                "dataset": "master_medicines_final",
                "record_id": str(self.source_row),
                "url": "https://www.nppaindia.nic.in",
                "caveat": None
                if self.provenance_complete
                else "gazette notification not recorded in source dataset",
            },
        }


@dataclass
class PriceCheck:
    """The result of comparing a product's price to its ceiling."""

    status: PriceStatus
    message: str
    market_price: float | None = None
    pack_count: float | None = None
    pack_unit: str | None = None
    market_price_per_unit: float | None = None
    ceiling_per_unit: float | None = None
    overcharge_per_unit: float | None = None
    overcharge_percent: float | None = None
    overcharge_per_pack: float | None = None
    ceiling: CeilingPrice | None = None
    workings: list[str] = field(default_factory=list)
    """The arithmetic, step by step, so the number can be checked by hand."""

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "message": self.message,
            "computed": self.status.startswith("verified"),
            "inputs": {
                "market_price": self.market_price,
                "pack_count": self.pack_count,
                "pack_unit": self.pack_unit,
            },
            "market_price_per_unit": self.market_price_per_unit,
            "ceiling_per_unit": self.ceiling_per_unit,
            "overcharge_per_unit": self.overcharge_per_unit,
            "overcharge_percent": self.overcharge_percent,
            "overcharge_per_pack": self.overcharge_per_pack,
            "workings": self.workings,
            "ceiling_source": self.ceiling.to_dict() if self.ceiling else None,
        }


class CeilingPriceTable:
    """NPPA ceiling prices, keyed by composition signature."""

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR):
        self._by_signature: dict[tuple, CeilingPrice] = {}
        self._coverage = {"rows": 0, "with_ceiling": 0, "with_notification": 0}
        self._load(Path(data_dir) / "master_medicines_final.csv")

    def _load(self, path: Path) -> None:
        if not path.exists():
            return

        with path.open(newline="", encoding="utf-8") as handle:
            for i, row in enumerate(csv.DictReader(handle)):
                self._coverage["rows"] += 1

                ceiling = parse_price(row.get("nppa_price_per_unit")) or parse_price(
                    row.get("nppa_price")
                )
                if ceiling is None:
                    continue

                ingredients = parse_python_literal_composition(row.get("composition_parsed", ""))
                signature = composition_signature(ingredients)
                if not signature:
                    continue

                self._coverage["with_ceiling"] += 1
                notification = (row.get("nppa_notif") or "").strip() or None
                if notification:
                    self._coverage["with_notification"] += 1

                pack = parse_pack_size(row.get("pack_size_label", ""))

                # Keep the lowest ceiling seen for a composition. Ceilings are a
                # legal maximum, so where the data offers several the strictest
                # is the safe one to hold a seller to.
                existing = self._by_signature.get(signature)
                if existing is None or ceiling < existing.ceiling_per_unit:
                    self._by_signature[signature] = CeilingPrice(
                        signature=signature,
                        ceiling_per_unit=ceiling,
                        unit=pack.unit,
                        source_row=i,
                        notification=notification,
                        notified_on=(row.get("nppa_date") or "").strip() or None,
                    )

    def lookup(self, signature: tuple) -> CeilingPrice | None:
        return self._by_signature.get(signature)

    @property
    def coverage(self) -> dict:
        """How much of the dataset actually carries a ceiling price.

        Reported in the API's `/v1/metrics` so the limitation is visible rather
        than inferred from an unexplained run of "no ceiling on record".
        """
        rows = max(self._coverage["rows"], 1)
        return {
            **self._coverage,
            "signatures": len(self._by_signature),
            "ceiling_coverage": round(self._coverage["with_ceiling"] / rows, 4),
            "notification_coverage": round(self._coverage["with_notification"] / rows, 4),
        }


def check_price(
    *,
    signature: tuple,
    market_price: float | None,
    pack_count: float | None,
    pack_unit: str,
    table: CeilingPriceTable,
) -> PriceCheck:
    """Compare a product's price against its NPPA ceiling.

    Returns a `PriceCheck` whose `workings` list contains every step, so the
    conclusion can be verified by hand rather than trusted.
    """
    if market_price is None:
        return PriceCheck(
            status="insufficient_data",
            message="No market price is recorded for this product.",
        )

    if not pack_count:
        return PriceCheck(
            status="insufficient_data",
            message=(
                "The pack size could not be read, so a per-unit price cannot be "
                "worked out. Comparing a whole-pack price against a per-tablet "
                "ceiling would be meaningless."
            ),
            market_price=market_price,
        )

    market_per_unit = round(market_price / pack_count, 4)
    workings = [
        f"Pack price Rs {market_price:.2f} / {pack_count:g} {pack_unit} "
        f"= Rs {market_per_unit:.4f} per {_singular(pack_unit)}"
    ]

    ceiling = table.lookup(signature)
    if ceiling is None:
        return PriceCheck(
            status="no_ceiling_on_record",
            message=(
                "This composition has no NPPA ceiling price on record. Most "
                "medicines are not under price control, so this is not a sign "
                "of anything wrong — but it also means the price cannot be "
                "checked against a legal maximum."
            ),
            market_price=market_price,
            pack_count=pack_count,
            pack_unit=pack_unit,
            market_price_per_unit=market_per_unit,
            workings=workings,
        )

    # Refuse to compare across units. A per-ml ceiling against a per-tablet
    # price produces a number, and the number is nonsense.
    if ceiling.unit != pack_unit:
        return PriceCheck(
            status="not_comparable",
            message=(
                f"The ceiling price is recorded per {_singular(ceiling.unit)} but this "
                f"pack is measured in {pack_unit}. These are not comparable, so no "
                "overcharge figure is given."
            ),
            market_price=market_price,
            pack_count=pack_count,
            pack_unit=pack_unit,
            market_price_per_unit=market_per_unit,
            ceiling_per_unit=ceiling.ceiling_per_unit,
            ceiling=ceiling,
            workings=workings,
        )

    ratio = market_per_unit / ceiling.ceiling_per_unit if ceiling.ceiling_per_unit else None

    if ratio is not None and ratio > IMPLAUSIBLE_RATIO:
        return PriceCheck(
            status="not_comparable",
            message=(
                f"The recorded price works out at {ratio:.0f} times the ceiling, which "
                "almost certainly means the two figures describe different pack sizes "
                "rather than a real overcharge. No conclusion is drawn."
            ),
            market_price=market_price,
            pack_count=pack_count,
            pack_unit=pack_unit,
            market_price_per_unit=market_per_unit,
            ceiling_per_unit=ceiling.ceiling_per_unit,
            ceiling=ceiling,
            workings=workings + [f"Ratio {market_per_unit:.4f} / "
                                 f"{ceiling.ceiling_per_unit:.4f} = {ratio:.1f}x — implausible"],
        )

    overcharge_per_unit = round(market_per_unit - ceiling.ceiling_per_unit, 4)
    overcharge_percent = round(
        (overcharge_per_unit / ceiling.ceiling_per_unit) * 100.0, 2
    ) if ceiling.ceiling_per_unit else None
    overcharge_per_pack = round(overcharge_per_unit * pack_count, 2)

    workings.append(
        f"NPPA ceiling Rs {ceiling.ceiling_per_unit:.4f} per {_singular(pack_unit)}"
    )
    workings.append(
        f"Difference Rs {market_per_unit:.4f} - Rs {ceiling.ceiling_per_unit:.4f} "
        f"= Rs {overcharge_per_unit:.4f} per {_singular(pack_unit)}"
    )
    workings.append(
        f"Over the whole pack: Rs {overcharge_per_unit:.4f} x {pack_count:g} "
        f"= Rs {overcharge_per_pack:.2f}"
    )

    over = overcharge_per_unit > 0
    if over:
        message = (
            f"This pack works out at Rs {market_per_unit:.2f} per {_singular(pack_unit)}, "
            f"which is Rs {overcharge_per_unit:.2f} ({overcharge_percent:.0f}%) above the "
            f"NPPA ceiling of Rs {ceiling.ceiling_per_unit:.2f}. That is "
            f"Rs {overcharge_per_pack:.2f} on this pack."
        )
    else:
        message = (
            f"This pack works out at Rs {market_per_unit:.2f} per {_singular(pack_unit)}, "
            f"within the NPPA ceiling of Rs {ceiling.ceiling_per_unit:.2f}."
        )

    if not ceiling.provenance_complete:
        message += (
            " Note: the source dataset does not record which gazette order set this "
            "ceiling, so the figure cannot be traced to its notification."
        )

    return PriceCheck(
        status="verified_over_ceiling" if over else "verified_within_ceiling",
        message=message,
        market_price=market_price,
        pack_count=pack_count,
        pack_unit=pack_unit,
        market_price_per_unit=market_per_unit,
        ceiling_per_unit=ceiling.ceiling_per_unit,
        overcharge_per_unit=overcharge_per_unit,
        overcharge_percent=overcharge_percent,
        overcharge_per_pack=overcharge_per_pack,
        ceiling=ceiling,
        workings=workings,
    )


def _singular(unit: str) -> str:
    return {"units": "unit", "ml": "ml", "g": "gram"}.get(unit, unit)


_TABLE: CeilingPriceTable | None = None


def get_ceiling_table(data_dir: Path = DEFAULT_DATA_DIR) -> CeilingPriceTable:
    global _TABLE
    if _TABLE is None:
        _TABLE = CeilingPriceTable(data_dir)
    return _TABLE
