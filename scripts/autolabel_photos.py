#!/usr/bin/env python3
"""
Auto-label real packaging photographs by running the full pipeline over them.

The point is to remove the manual transcription step from building an eval set.
An image whose OCR text resolves to a composition *confidently* is one where
the ground truth is known without a human reading the strip — the resolver had
to match against 253,973 real products to get there, so a confident hit is
strong evidence, not a guess.

Images that do not resolve are equally valuable: they are the hard cases, and
they are exactly where a human label is worth paying for. This script tells you
which handful of images that is, instead of transcribing all of them.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402

from packages.perception import tesseract_engine as te  # noqa: E402
from packages.perception.dip import acquire  # noqa: E402
from packages.perception.dip.pipeline import run_auto  # noqa: E402
from packages.resolver.calibrate import load_or_default  # noqa: E402
from packages.resolver.index import get_index  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=pathlib.Path,
                        default=REPO_ROOT / "data" / "raw" / "packaging" / "images")
    parser.add_argument("--out", type=pathlib.Path,
                        default=REPO_ROOT / "data" / "raw" / "packaging" / "autolabel.json")
    parser.add_argument("--max-dimension", type=int, default=1400,
                        help="downscale before processing; Non-Local Means is O(n^2) "
                             "in a way that makes a 4000px source take minutes")
    args = parser.parse_args(argv)

    index = get_index()
    calibrator = load_or_default()

    images = sorted(p for p in args.dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(f"processing {len(images)} images", flush=True)

    results = []
    started = time.perf_counter()

    for i, path in enumerate(images, 1):
        try:
            # Downscale first. The full-resolution originals are up to 4000px
            # and the adaptive denoiser picks Non-Local Means on them, which
            # turns a 2-second job into a 90-second one for no accuracy gain —
            # the DIP pipeline caps at 2000px internally anyway.
            image = acquire.decode(path.read_bytes())
            image, _ = acquire.limit_resolution(image, args.max_dimension)

            dip = run_auto(image)
            ocr = te.read_renditions(dip.renditions)
            tokens = ocr.consensus_tokens or ocr.tokens
            query = " ".join(tokens)

            matches = (
                index.search_compositions_from_tokens(
                    tokens, strengths=ocr.strengths, top_k=3
                )
                if query.strip()
                else []
            )
            status, probability = calibrator.decide(matches, query)

            results.append(
                {
                    "file": path.name,
                    "quality": dip.quality.verdict,
                    "preset": dip.metrics.get("auto_preset"),
                    "glare": round(dip.quality.glare_fraction, 3),
                    "blur": round(dip.quality.blur_variance, 1),
                    "boundary": dip.metrics.get("boundary_method"),
                    "rectified": dip.metrics.get("rectified"),
                    "ocr_confidence": round(ocr.mean_confidence, 1),
                    "n_tokens": len(tokens),
                    "tokens": tokens[:20],
                    "ocr_text": ocr.text[:300],
                    "status": status,
                    "probability": round(probability, 4),
                    "composition": matches[0].label if matches else None,
                    "closest_brand": matches[0].best_name if matches else None,
                    "signature": list(matches[0].signature) if matches else None,
                }
            )
        except Exception as exc:  # noqa: BLE001
            results.append({"file": path.name, "error": f"{type(exc).__name__}: {exc}"[:200]})

        if i % 10 == 0 or i == len(images):
            elapsed = time.perf_counter() - started
            print(f"  {i}/{len(images)}  ({elapsed:.0f}s, {elapsed / i:.1f}s each)", flush=True)

    args.out.write_text(json.dumps(results, indent=2))

    import collections

    ok = [r for r in results if "error" not in r]
    confident = [r for r in ok if r["status"] == "confident"]

    print()
    print(f"quality  : {collections.Counter(r['quality'] for r in ok).most_common()}")
    print(f"status   : {collections.Counter(r['status'] for r in ok).most_common()}")
    print(f"errors   : {len(results) - len(ok)}")
    print(f"\nCONFIDENTLY RESOLVED: {len(confident)}/{len(ok)} "
          f"({len(confident) / max(len(ok), 1):.0%}) — these are auto-labelled")
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
