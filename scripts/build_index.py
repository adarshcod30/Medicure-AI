#!/usr/bin/env python3
"""
Build the resolver index artifacts from the CSVs in data/processed/.

Run this once after cloning, and again whenever a dataset or the normalisation
logic changes:

    python scripts/build_index.py

Artifacts land in data/artifacts/ and are gitignored — they are derived data,
rebuildable in about a minute, and large enough that committing them would
bloat the repository for no benefit.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from packages.resolver.index import (  # noqa: E402
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_DATA_DIR,
    build,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    required = ["A_Z_medicines_dataset_of_India.csv", "generic.csv"]
    missing = [name for name in required if not (args.data_dir / name).exists()]
    if missing:
        print(f"error: missing dataset(s) in {args.data_dir}: {', '.join(missing)}",
              file=sys.stderr)
        return 1

    started = time.perf_counter()
    stats = build(args.data_dir, args.artifact_dir, verbose=not args.quiet)
    elapsed = time.perf_counter() - started

    print()
    print(f"  brands indexed            {stats['brands']:>9,}")
    print(f"  Jan Aushadhi products     {stats['generics']:>9,}")
    print(f"  distinct brand signatures {stats['brand_signatures']:>9,}")
    print(f"  substitutable signatures  {stats['substitutable_signatures']:>9,}")
    print(f"  brands per composition    {stats['brands_per_signature']:>9.1f}")
    print(f"  name features             {stats['name_features']:>9,}")
    print(f"  composition features      {stats['composition_features']:>9,}")
    print(f"  matrix size               {stats['matrix_mb']:>9.1f} MB")
    print(f"  build time                {elapsed:>9.1f} s")
    print(f"\nartifacts written to {args.artifact_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
