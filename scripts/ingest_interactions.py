#!/usr/bin/env python3
"""
Convert DDInter release CSVs into the interaction table this project reads.

DDInter is a curated drug-drug interaction database (Xiong et al., Nucleic
Acids Research 2022), free for academic use. It is NOT downloaded
automatically — fetch the release files yourself and place them in
data/raw/ddinter/:

    https://ddinter.scbdd.com/download/

Download the per-category CSVs (ddinter_downloads_code_A.csv, _B.csv, ...) and
drop them in that directory. This script reads whatever is there.

What it does beyond reformatting: it maps DDInter's drug names into the
canonical ingredient vocabulary this project's index uses, and it writes ONLY
the rows it could map. An unmapped row is worse than a missing one — the
lookup would never hit it, so it would inflate the apparent coverage while
contributing nothing. The match rate is reported so the gap is visible.

    python scripts/ingest_interactions.py
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from packages.resolver.index import DEFAULT_ARTIFACT_DIR, get_index  # noqa: E402
from packages.resolver.normalize import canonical_ingredient  # noqa: E402

RAW_DIR = REPO_ROOT / "data" / "raw" / "ddinter"
OUT_PATH = REPO_ROOT / "data" / "processed" / "interactions" / "interactions.csv"
SOURCE_URL = "https://ddinter.scbdd.com/"

DDINTER_ALIASES = {
    # DDInter labels drugs with USAN (American) names; Indian packaging uses
    # BAN/INN (British) names. Same molecule, different naming convention.
    # Without these, 4,770 interaction mentions were silently dropped --
    # including every interaction involving adrenaline, which is not a gap
    # worth having.
    #
    # This table lives here, in the ingest, rather than in
    # resolver/normalize.py on purpose. Editing canonical_ingredient would
    # shift every composition signature and invalidate the index AND the
    # calibrator (see CLAUDE.md). This is a problem of reconciling one
    # dataset's vocabulary with ours, not a change to what a composition
    # signature means, so it belongs to the dataset.
    #
    # Every entry is a naming-convention synonym for the SAME molecule.
    # Deliberately NOT included: prednisone -> prednisolone. They are
    # different drugs in a prodrug relationship, and mapping them would
    # invent a clinical equivalence rather than translate a name. Roughly
    # 1,869 prednisone mentions stay dropped, correctly.
    "epinephrine": "adrenaline",
    "glyburide": "glibenclamide",
    "rifampin": "rifampicin",
    "ethanol": "alcohol",
    "levothyroxine": "thyroxine",
    "meperidine": "pethidine",
    "isoproterenol": "isoprenaline",
    "cromolyn": "sodium cromoglycate",
    "phenobarbital": "phenobarbitone",
    "benzylpenicillin": "penicillin g",
}

INSTRUCTIONS = f"""
No DDInter files found in {RAW_DIR}

The interaction engine needs a dataset; it will not invent one, so this step
is required before /v1/interactions and the cabinet's interaction panel report
anything.

  1. Open https://ddinter.scbdd.com/download/
  2. Download the category CSVs (ddinter_downloads_code_A.csv, _B.csv, _D.csv,
     _H.csv, _L.csv, _P.csv, _R.csv, _V.csv)
  3. Put them in:  {RAW_DIR}
  4. Re-run:       python scripts/ingest_interactions.py

Expected columns per file: DDInterID_A, Drug_A, DDInterID_B, Drug_B, Level.
The script tolerates extra columns and different orderings.

Until then the engine reports itself unavailable, which is the correct
behaviour: an empty interaction list with an honest "no dataset installed"
note, rather than a reassuring silence.
"""


def _map(name: str) -> str:
    """Fold a DDInter drug name into our canonical vocabulary.

    Alias first, then the shared canonical fold, so an alias target still gets
    salt-stripped exactly like every other ingredient in the system.
    """
    folded = canonical_ingredient(name)
    return canonical_ingredient(DDINTER_ALIASES.get(folded, folded))


def catalogue_vocabulary() -> set[str]:
    """Every canonical ingredient the index actually knows about."""
    index = get_index(DEFAULT_ARTIFACT_DIR)
    vocabulary: set[str] = set()
    for signature in index._signatures:  # noqa: SLF001 — build script
        for component in signature or ():
            vocabulary.add(canonical_ingredient(str(component[0])))
    for generic in index.all_generics():
        for component in generic.signature or ():
            vocabulary.add(canonical_ingredient(str(component[0])))
    vocabulary.discard("")
    return vocabulary


def main() -> int:
    files = sorted(RAW_DIR.glob("*.csv")) if RAW_DIR.exists() else []
    if not files:
        print(INSTRUCTIONS)
        return 0

    print(f"reading {len(files)} DDInter file(s) from {RAW_DIR}")
    vocabulary = catalogue_vocabulary()
    print(f"catalogue vocabulary: {len(vocabulary)} canonical ingredients")

    rows: dict[tuple[str, str], dict] = {}
    total = 0
    unmapped: Counter = Counter()

    for path in files:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                total += 1
                lower = {k.strip().lower(): (v or "").strip() for k, v in row.items()}
                drug_a = lower.get("drug_a") or lower.get("drug a") or ""
                drug_b = lower.get("drug_b") or lower.get("drug b") or ""
                level = lower.get("level") or lower.get("severity") or "Unknown"

                a = _map(drug_a)
                b = _map(drug_b)
                if not a or not b or a == b:
                    continue

                # Both sides must exist in the catalogue, or the pair can never
                # be reached from a scan and only inflates apparent coverage.
                if a not in vocabulary:
                    unmapped[drug_a.lower()] += 1
                if b not in vocabulary:
                    unmapped[drug_b.lower()] += 1
                if a not in vocabulary or b not in vocabulary:
                    continue

                key = (a, b) if a <= b else (b, a)
                rows[key] = {
                    "ingredient_a": key[0],
                    "ingredient_b": key[1],
                    "severity": level.title(),
                    # DDInter's public release carries a severity level, not a
                    # prose mechanism. Writing a description here would mean
                    # generating one, so the field stays empty and the UI shows
                    # the severity and the citation instead.
                    "description": "",
                    "ddinter_id_a": lower.get("ddinterid_a", ""),
                    "ddinter_id_b": lower.get("ddinterid_b", ""),
                    "source_url": SOURCE_URL,
                }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "ingredient_a", "ingredient_b", "severity", "description",
                "ddinter_id_a", "ddinter_id_b", "source_url",
            ],
        )
        writer.writeheader()
        for key in sorted(rows):
            writer.writerow(rows[key])

    rate = len(rows) / total if total else 0.0
    print(f"\nread            : {total} rows")
    print(f"written         : {len(rows)} unique pairs ({rate:.1%} of input)")
    print(f"output          : {OUT_PATH}")
    if unmapped:
        print(f"\nunmapped drug names (top 15 of {len(unmapped)}):")
        for name, count in unmapped.most_common(15):
            print(f"  {count:5}  {name}")
        print(
            "\nThese are drugs DDInter covers that this catalogue's vocabulary "
            "does not. They are dropped deliberately: a pair that cannot be "
            "reached from a scan would inflate coverage without adding safety."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
