#!/usr/bin/env python3
"""
Fit the abstention calibrator and write data/artifacts/calibrator.joblib.

Requires the index to exist (run scripts/build_index.py first).

    python scripts/fit_calibrator.py --products 3000 --target-precision 0.95

Training data is synthetic: real product names and compositions corrupted with
a character confusion matrix derived from how OCR actually fails. Refit against
the real photo set once it exists — the thresholds this produces are only as
representative as the corruption model behind them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from packages.resolver.calibrate import fit  # noqa: E402
from packages.resolver.index import DEFAULT_ARTIFACT_DIR, get_index  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--products", type=int, default=3000)
    parser.add_argument("--target-precision", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    args = parser.parse_args(argv)

    try:
        index = get_index(args.artifact_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    started = time.perf_counter()
    calibrator = fit(
        index,
        n_products=args.products,
        target_precision=args.target_precision,
        seed=args.seed,
    )

    out = args.artifact_dir / "calibrator.joblib"
    calibrator.save(out)

    report_path = args.artifact_dir / "calibration_report.json"
    if calibrator.report:
        report_path.write_text(json.dumps(calibrator.report.to_dict(), indent=2))

    print(f"\nfitted in {time.perf_counter() - started:.1f}s")
    print(f"calibrator -> {out}")
    print(f"report     -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
