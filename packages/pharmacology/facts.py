"""
What a medicine treats, and what it can do to you.

The two fields users ask for first — "what is this for?" and "what are the side
effects?" — and the two the system this replaced was worst at, because it asked
a language model and got fluent, uncheckable answers. A hallucinated indication
is not a cosmetic error: it is someone taking an antihistamine for chest pain.

So the same rule as every other package here: these come from a dataset row
with a citation, or they do not come at all. There is no code path that can
generate an indication, and when the dataset is absent the engine says so
rather than degrading into invention.

Keyed by composition signature rather than brand, because the molecule is what
determines the pharmacology. Every one of the twenty-four brands selling
paracetamol 500mg has the same indications, and storing them per brand would
mean twenty-four chances to disagree with itself.
"""

from __future__ import annotations

import ast
import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
FACTS_FILE = "facts/medicine_facts.csv"


@dataclass
class MedicineFacts:
    """Clinical facts for one composition, exactly as the dataset recorded them."""

    uses: list[str] = field(default_factory=list)
    side_effects: list[str] = field(default_factory=list)
    therapeutic_class: str = ""
    habit_forming: str = ""
    source: dict = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return bool(self.uses or self.side_effects)

    def to_dict(self) -> dict:
        return {
            "uses": self.uses,
            "side_effects": self.side_effects,
            "therapeutic_class": self.therapeutic_class or None,
            "habit_forming": self.habit_forming or None,
            "source": self.source,
            # Said explicitly rather than implied by empty lists. "We have no
            # record" and "this medicine has no side effects" are opposite
            # claims, and an empty array alone reads as the second.
            "note": (
                "Recorded from the source dataset. Absence here means the "
                "dataset has no entry, not that none exist."
                if self.available
                else "No clinical record for this composition in the installed dataset."
            ),
        }


class FactsTable:
    """Composition -> clinical facts. Absent data is a reported state."""

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR):
        self.path = Path(data_dir) / FACTS_FILE
        self._by_signature: dict[tuple, MedicineFacts] = {}
        self.available = False
        self.last_error: str | None = None
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self.last_error = (
                f"{self.path} not found. Build it with: "
                "python scripts/ingest_medicine_facts.py"
            )
            logger.warning("medicine facts unavailable: %s", self.last_error)
            return

        try:
            with self.path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    try:
                        signature = ast.literal_eval(row.get("composition_sig", ""))
                    except (ValueError, SyntaxError):
                        continue
                    if not isinstance(signature, tuple):
                        continue

                    self._by_signature[signature] = MedicineFacts(
                        uses=_split(row.get("uses")),
                        side_effects=_split(row.get("side_effects")),
                        therapeutic_class=(row.get("therapeutic_class") or "").strip(),
                        habit_forming=(row.get("habit_forming") or "").strip(),
                        source={
                            "dataset": (row.get("source_dataset") or "").strip(),
                            "record_id": (row.get("source_brand") or "").strip(),
                            "url": None,
                        },
                    )
            self.available = True
        except (OSError, csv.Error) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("medicine facts unreadable: %s", exc)

    def __len__(self) -> int:
        return len(self._by_signature)

    def lookup(self, signature: tuple) -> MedicineFacts:
        """Facts for a composition, or an empty record. Never None."""
        return self._by_signature.get(signature) or MedicineFacts()

    def status(self) -> dict:
        return {
            "available": self.available,
            "compositions": len(self._by_signature),
            "path": str(self.path),
            "error": self.last_error,
        }


def _split(value: str | None) -> list[str]:
    """The ingest joins multi-valued fields with ' | '."""
    if not value:
        return []
    return [part.strip() for part in value.split("|") if part.strip()]


_table: FactsTable | None = None


def get_facts_table(data_dir: Path = DEFAULT_DATA_DIR) -> FactsTable:
    global _table
    if _table is None:
        _table = FactsTable(data_dir)
    return _table
