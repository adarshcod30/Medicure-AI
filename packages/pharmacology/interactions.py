"""
Drug-drug interaction checking.

The hardest rule in this project applies here more than anywhere:

    Never let a language model produce an interaction.

Every finding in this module comes from a dataset row and carries the row's
identifiers back to the caller. There is no code path that can invent one, and
no field in the response for a claim without a `source`. If the dataset file
is absent the engine reports itself unavailable — it does not fall back to
anything, because the only available fallback would be fabrication.

Ingredient names are folded with `resolver.normalize.canonical_ingredient`,
the same function the index uses. That is not a detail: the dataset's
vocabulary and the catalogue's vocabulary must meet at the same canonical
form, or every lookup silently misses and the engine reports "no interactions"
for a pair it holds data on. Silent under-reporting is the dangerous failure
here, so `scripts/ingest_interactions.py` reports its match rate and refuses
to write rows it could not map.

Duplicate-therapy detection needs no dataset at all: the same folded
ingredient appearing in two different products is arithmetic over what the
resolver already determined.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from packages.resolver.normalize import canonical_ingredient

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
INTERACTIONS_FILE = "interactions/interactions.csv"

SEVERITY_ORDER = {"Major": 0, "Moderate": 1, "Minor": 2, "Unknown": 3}
"""Sort key. Major first — if a list gets truncated in a UI, the truncation
must not be what hides the serious finding."""


@dataclass
class Interaction:
    """One dataset row, carried through unchanged."""

    ingredient_a: str
    ingredient_b: str
    severity: str
    description: str
    source: dict

    def to_dict(self) -> dict:
        return {
            "ingredients": [self.ingredient_a, self.ingredient_b],
            "severity": self.severity,
            "description": self.description,
            "source": self.source,
        }


@dataclass
class DuplicateTherapy:
    """The same active ingredient reached by two different products."""

    ingredient: str
    items: list[str]

    def to_dict(self) -> dict:
        return {
            "ingredient": self.ingredient,
            "items": self.items,
            "message": (
                f"{len(self.items)} of these products contain {self.ingredient}. "
                "Taking them together means taking that ingredient more than "
                "once, which can exceed the intended dose."
            ),
        }


class InteractionTable:
    """Pairwise interactions keyed by folded ingredient pair.

    Absent data is a reported state, not an exception — the same contract as
    MongoStore and the Bedrock client. A deployment without the dataset serves
    every other feature and says plainly that this one is off.
    """

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR):
        self.path = Path(data_dir) / INTERACTIONS_FILE
        self._pairs: dict[tuple[str, str], Interaction] = {}
        self.available = False
        self.last_error: str | None = None
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self.last_error = (
                f"{self.path} not found. Build it with: "
                "python scripts/ingest_interactions.py"
            )
            logger.warning("interaction data unavailable: %s", self.last_error)
            return

        try:
            with self.path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    a = canonical_ingredient(row.get("ingredient_a", ""))
                    b = canonical_ingredient(row.get("ingredient_b", ""))
                    if not a or not b or a == b:
                        continue
                    severity = (row.get("severity") or "Unknown").strip().title()
                    if severity not in SEVERITY_ORDER:
                        severity = "Unknown"
                    self._pairs[_key(a, b)] = Interaction(
                        ingredient_a=a,
                        ingredient_b=b,
                        severity=severity,
                        description=(row.get("description") or "").strip(),
                        source={
                            "dataset": "ddinter",
                            "record_id": f"{row.get('ddinter_id_a', '')}|{row.get('ddinter_id_b', '')}",
                            "url": (row.get("source_url") or "").strip() or None,
                        },
                    )
            self.available = True
        except (OSError, csv.Error) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("interaction data unreadable: %s", exc)

    def __len__(self) -> int:
        return len(self._pairs)

    def lookup(self, a: str, b: str) -> Interaction | None:
        """Order-independent pair lookup on already-folded names."""
        return self._pairs.get(_key(canonical_ingredient(a), canonical_ingredient(b)))

    def status(self) -> dict:
        return {
            "available": self.available,
            "pairs": len(self._pairs),
            "path": str(self.path),
            "error": self.last_error,
        }


def _key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def ingredients_of(signature: tuple) -> list[str]:
    """Folded ingredient names from a composition signature."""
    return [canonical_ingredient(str(component[0])) for component in signature if component]


def check_signatures(
    signatures: list[tuple],
    *,
    labels: list[str] | None = None,
    table: InteractionTable | None = None,
) -> dict:
    """Check a set of products against each other.

    `labels` names the products for the response; it defaults to positional
    names. Returns findings, duplicate therapy, and an explicit coverage note —
    an empty `findings` list means "nothing on record", never "safe", and the
    note says so.
    """
    table = table if table is not None else get_interaction_table()
    names = labels or [f"item {i + 1}" for i in range(len(signatures))]

    per_item = [ingredients_of(signature) for signature in signatures]

    findings: list[Interaction] = []
    checked_pairs = 0
    if table.available:
        seen: set[tuple[str, str]] = set()
        for i, j in combinations(range(len(signatures)), 2):
            for a in per_item[i]:
                for b in per_item[j]:
                    if a == b:
                        continue
                    checked_pairs += 1
                    key = _key(a, b)
                    if key in seen:
                        continue
                    interaction = table.lookup(a, b)
                    if interaction is not None:
                        seen.add(key)
                        findings.append(interaction)

    findings.sort(key=lambda f: (SEVERITY_ORDER.get(f.severity, 3), f.ingredient_a))

    # Duplicate therapy is arithmetic, so it works with no dataset at all.
    where: dict[str, list[str]] = {}
    for name, ingredients in zip(names, per_item):
        for ingredient in set(ingredients):
            where.setdefault(ingredient, []).append(name)
    duplicates = [
        DuplicateTherapy(ingredient=ingredient, items=items)
        for ingredient, items in sorted(where.items())
        if len(items) > 1
    ]

    if not table.available:
        note = (
            "No interaction dataset is installed on this deployment, so no "
            "interaction check was performed. Duplicate-ingredient checking is "
            "unaffected because it needs no dataset."
        )
    elif not findings:
        note = (
            "No interactions on record between these items. The dataset's "
            "coverage is partial, so this is an absence of evidence rather "
            "than evidence of safety — ask a pharmacist about any combination "
            "you are unsure of."
        )
    else:
        note = (
            f"{len(findings)} interaction(s) found on record. The dataset's "
            "coverage is partial, so absence of a further finding is not "
            "evidence of safety."
        )

    return {
        "available": table.available,
        "findings": [f.to_dict() for f in findings],
        "duplicate_therapy": [d.to_dict() for d in duplicates],
        "items_checked": len(signatures),
        "ingredient_pairs_checked": checked_pairs,
        "coverage_note": note,
        "dataset": table.status(),
    }


_table: InteractionTable | None = None


def get_interaction_table(data_dir: Path = DEFAULT_DATA_DIR) -> InteractionTable:
    global _table
    if _table is None:
        _table = InteractionTable(data_dir)
    return _table
