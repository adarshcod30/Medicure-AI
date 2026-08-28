#!/usr/bin/env python3
"""
Ingest per-medicine clinical facts: what it treats, its side effects, its class.

The A-Z catalogue carries nine columns and none of them say what a medicine is
FOR. That gap is why the system this replaced asked a language model — and why
it produced fluent, uncheckable indications. Facts of this kind have to come
from a dataset with a citation or not at all.

Source: the Kaggle "Medicine Dataset" (~250k Indian medicines), which carries
use0..4, sideEffect0..41, substitute0..4 and therapeutic class, keyed on brand
names that overlap the A-Z catalogue.

    mkdir -p data/raw/medicine_uses && <put the CSV there>
    python scripts/ingest_medicine_facts.py

Design mirrors ingest_interactions.py, for the same reasons:

- Rows are keyed by COMPOSITION SIGNATURE, not brand name. Identity in this
  system is the molecule; two brands of paracetamol 500mg should not need two
  separate rows of side effects, and a brand absent from the catalogue would
  otherwise contribute facts nothing can reach.
- Only rows whose brand resolves into the catalogue are written, and the match
  rate is printed. An unreachable row inflates apparent coverage while adding
  nothing.
- Nothing is generated. If the source has no side effects for a product, the
  field stays empty and the API says so rather than filling it in.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from packages.resolver.index import DEFAULT_ARTIFACT_DIR, get_index  # noqa: E402
from packages.resolver.normalize import composition_signature  # noqa: E402

RAW_DIR = REPO_ROOT / "data" / "raw" / "medicine_uses"
OUT_PATH = REPO_ROOT / "data" / "processed" / "facts" / "medicine_facts.csv"
SOURCE = "kaggle_medicine_dataset"

INSTRUCTIONS = f"""
No source file found in {RAW_DIR}

This dataset supplies what a medicine TREATS and its SIDE EFFECTS. Without it
the API reports those fields as unavailable rather than inventing them, so
this step is required before /v1/scan returns uses or side effects.

  1. Download the Kaggle "Medicine Dataset" (~250k Indian medicines, with
     use0..4, sideEffect0..41, substitute0..4, Therapeutic Class)
  2. Put the CSV in:  {RAW_DIR}
  3. Re-run:          python scripts/ingest_medicine_facts.py

Any CSV in that directory is read. Column names are detected case-insensitively
and the script tolerates the several spellings this dataset ships with.
"""


def _columns(header: list[str], *prefixes: str) -> list[str]:
    """All columns whose lowercased name starts with any given prefix."""
    return [c for c in header if c and c.strip().lower().startswith(prefixes)]


def _values(row: dict, columns: list[str]) -> list[str]:
    seen: list[str] = []
    for c in columns:
        v = (row.get(c) or "").strip()
        if v and v.lower() not in {"nan", "none", "null"} and v not in seen:
            seen.append(v)
    return seen


def main() -> int:
    files = sorted(RAW_DIR.glob("*.csv")) if RAW_DIR.exists() else []
    if not files:
        print(INSTRUCTIONS)
        return 0

    index = get_index(DEFAULT_ARTIFACT_DIR)
    print(f"reading {len(files)} file(s) from {RAW_DIR}")

    rows: dict[tuple, dict] = {}
    total = 0
    unmatched: Counter = Counter()

    for path in files:
        with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
            reader = csv.DictReader(handle)
            header = reader.fieldnames or []
            name_col = next(
                (c for c in header if c and c.strip().lower() in {"name", "medicine", "brand"}),
                None,
            )
            if not name_col:
                print(f"  skipping {path.name}: no name column in {header[:6]}", file=sys.stderr)
                continue

            use_cols = _columns(header, "use")
            effect_cols = _columns(header, "sideeffect", "side_effect")
            class_cols = _columns(header, "therapeutic", "chemical class", "action class")
            habit_col = next((c for c in header if "habit" in c.lower()), None)
            print(f"  {path.name}: {len(use_cols)} use cols, {len(effect_cols)} side-effect cols")

            for row in reader:
                total += 1
                brand = (row.get(name_col) or "").strip()
                if not brand:
                    continue

                # Resolve the brand into the catalogue, then key by its
                # composition. A name the index cannot place is dropped: its
                # facts could never be reached from a scan.
                matches = index.search(brand, top_k=1, min_similarity=0.75)
                if not matches:
                    unmatched[brand.lower()] += 1
                    continue
                signature = matches[0].signature
                if not signature:
                    continue

                uses = _values(row, use_cols)
                effects = _values(row, effect_cols)
                if not uses and not effects:
                    continue

                # Keep the richest row per composition; different brands of the
                # same molecule describe it with differing completeness.
                existing = rows.get(signature)
                score = len(uses) + len(effects)
                if existing and existing["_score"] >= score:
                    continue

                rows[signature] = {
                    "_score": score,
                    "composition_sig": repr(signature),
                    "example_brand": matches[0].name,
                    "uses": " | ".join(uses),
                    "side_effects": " | ".join(effects),
                    "therapeutic_class": " | ".join(_values(row, class_cols)),
                    "habit_forming": (row.get(habit_col) or "").strip() if habit_col else "",
                    "source_dataset": SOURCE,
                    "source_brand": brand,
                }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields = ["composition_sig", "example_brand", "uses", "side_effects",
              "therapeutic_class", "habit_forming", "source_dataset", "source_brand"]
    with OUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for signature in sorted(rows, key=repr):
            writer.writerow({k: v for k, v in rows[signature].items() if k in fields})

    rate = len(rows) / total if total else 0.0
    print(f"\nread              : {total} source rows")
    print(f"written           : {len(rows)} compositions ({rate:.1%} of input)")
    print(f"output            : {OUT_PATH}")
    if unmatched:
        print(f"\nbrands the catalogue could not place (top 10 of {len(unmatched)}):")
        for name, count in unmatched.most_common(10):
            print(f"  {count:5}  {name}")
        print(
            "\nDropped on purpose: a composition the index cannot resolve is one no "
            "scan can ever reach, so keeping it would inflate coverage without "
            "adding a single answerable question."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
