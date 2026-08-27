#!/usr/bin/env python3
"""
Benchmark on real packaging photographs, with labels obtained for free.

The problem this solves: a benchmark needs (image, ground truth) pairs, and
transcribing hundreds of medicine strips by hand is the reason such datasets do
not exist for the Indian market.

The trick is that the label is already written down next to the image. Retail
listings (Amazon.in, PharmEasy, 1mg) caption their product photos with the
brand name — "Zenflox-DT 100 - Strip of 10 Tablets". That caption is not the
answer we are testing for; the answer is the *composition*, which the caption
does not state. So:

    caption -> brand name -> look up in the 253,973-row index -> composition

gives ground truth that no human had to read off the strip, and the image
itself is never involved in producing it. The pipeline then reads the image and
its answer is compared. The two paths share nothing but the product.

The labelling step is verified rather than assumed: a caption whose brand does
not resolve above `--label-threshold` is discarded, not guessed at. On the first
sample of eight, all eight resolved above 0.67.

    python -m eval.bench_real_images --dir data/raw/gimages

The harvest itself is NOT in the repository: it is scraped retail imagery, and
redistributing it is not ours to do. `--dir` must point at a directory holding
`harvest.tsv` (caption per image) and `images/`. Without it the script exits
with a message naming both expected paths rather than pretending to run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from packages.perception import tesseract_engine as te  # noqa: E402
from packages.perception.dip import acquire  # noqa: E402
from packages.perception.dip.pipeline import run_auto  # noqa: E402
from packages.resolver.calibrate import load_or_default  # noqa: E402
from packages.resolver.index import get_index  # noqa: E402

# Retail caption boilerplate. Stripped so the brand name is what gets looked up.
CAPTION_NOISE = [
    " - Strip of", " Strip Of", " - strip of", ": Amazon.in", "Amazon.in",
    "Health & Personal Care", "| PharmEasy", "PharmEasy", "- National Pharmacy",
    "Price - Buy Online at Best Price in India", "Buy Online", "Uses, Side Effects",
    "Price & Dosage", "Tablets -", "Blister Pack",
]


def brand_from_caption(caption: str) -> str:
    """Extract the brand name from a retail caption.

    Everything from the first separator onward is pack description and retailer
    boilerplate, not identity.
    """
    text = caption
    for token in CAPTION_NOISE:
        idx = text.find(token)
        if idx > 0:
            text = text[:idx]
    for sep in (" - ", " – ", ": ", " | ", " Strip", " strip"):
        idx = text.find(sep)
        if idx > 2:
            text = text[:idx]
    return text.strip(" -–:|,")


def build_labels(index, harvest: Path, threshold: float) -> dict[str, dict]:
    """Caption -> verified ground-truth composition, keyed by image filename."""
    labels: dict[str, dict] = {}
    rejected = 0

    for line in harvest.read_text().splitlines():
        if "\t" not in line:
            continue
        caption, _url = line.split("\t", 1)
        brand = brand_from_caption(caption)
        if len(brand) < 3:
            rejected += 1
            continue

        matches = index.search(brand, top_k=1)
        if not matches or matches[0].similarity < threshold:
            rejected += 1
            continue

        filename = hashlib.md5(caption.encode()).hexdigest()[:12] + ".jpg"
        labels[filename] = {
            "caption": caption,
            "brand_from_caption": brand,
            "matched_product": matches[0].name,
            "label_similarity": round(matches[0].similarity, 3),
            "truth_signature": matches[0].signature,
            "truth_composition": matches[0].composition,
        }

    return labels, rejected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=REPO_ROOT / "data" / "raw" / "gimages")
    parser.add_argument("--label-threshold", type=float, default=0.55)
    parser.add_argument("--max-dimension", type=int, default=1400)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "eval" / "results")
    args = parser.parse_args(argv)

    harvest = args.dir / "harvest.tsv"
    images_dir = args.dir / "images"
    if not harvest.exists() or not images_dir.exists():
        print(f"error: expected {harvest} and {images_dir}", file=sys.stderr)
        return 1

    index = get_index()
    calibrator = load_or_default()

    labels, rejected = build_labels(index, harvest, args.label_threshold)
    print(f"labels: {len(labels)} verified, {rejected} captions rejected "
          f"(brand did not resolve above {args.label_threshold})")

    available = {p.name for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"}}
    usable = sorted(available & set(labels))
    print(f"images: {len(available)} on disk, {len(usable)} with a verified label\n")

    rows = []
    started = time.perf_counter()

    for name in usable:
        label = labels[name]
        try:
            image = acquire.decode((images_dir / name).read_bytes())
            image, _ = acquire.limit_resolution(image, args.max_dimension)

            dip = run_auto(image)
            ocr = te.read_renditions(dip.renditions)
            tokens = ocr.consensus_tokens or ocr.tokens
            query = " ".join(tokens)

            matches = (
                index.search_compositions_from_tokens(tokens, strengths=ocr.strengths, top_k=5)
                if query.strip()
                else []
            )
            status, probability = calibrator.decide(matches, query)

            correct = bool(matches) and matches[0].signature == label["truth_signature"]
            in_top5 = any(m.signature == label["truth_signature"] for m in matches)

            rows.append(
                {
                    "file": name,
                    "brand": label["brand_from_caption"],
                    "truth": label["truth_composition"],
                    "predicted": matches[0].label if matches else None,
                    "correct": correct,
                    "in_top5": in_top5,
                    "status": status,
                    "probability": round(probability, 3),
                    "quality": dip.quality.verdict,
                    "preset": dip.metrics.get("auto_preset"),
                    "ocr_confidence": round(ocr.mean_confidence, 1),
                    "n_tokens": len(tokens),
                    "tokens": tokens[:12],
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append({"file": name, "error": f"{type(exc).__name__}: {exc}"[:150]})

    elapsed = time.perf_counter() - started
    ok = [r for r in rows if "error" not in r]
    if not ok:
        print("no images processed successfully", file=sys.stderr)
        return 1

    answered = [r for r in ok if r["status"] in {"confident", "ambiguous"}]
    correct = [r for r in ok if r["correct"]]
    confidently_wrong = [r for r in ok if r["status"] == "confident" and not r["correct"]]

    print("=" * 74)
    print("REAL PACKAGING PHOTOGRAPHS — Indian retail listings")
    print("=" * 74)
    print(f"  images                    {len(ok)}")
    print(f"  top-1 composition correct {len(correct) / len(ok):.1%}")
    print(f"  top-5 composition correct {sum(r['in_top5'] for r in ok) / len(ok):.1%}")
    print(f"  answered                  {len(answered) / len(ok):.1%}")
    print(f"  SILENT FAILURE            {len(confidently_wrong) / len(ok):.1%}")
    print(f"  mean OCR confidence       {sum(r['ocr_confidence'] for r in ok) / len(ok):.1f}")
    print(f"  seconds per image         {elapsed / len(ok):.1f}")
    print()

    for r in sorted(ok, key=lambda x: (not x["correct"], -x["probability"])):
        mark = "OK  " if r["correct"] else ("top5" if r["in_top5"] else "MISS")
        print(f"  [{mark}] {r['brand'][:18]:18s} P={r['probability']:.2f} {r['status'][:10]:10s} "
              f"truth={r['truth'][:30]:30s} got={str(r['predicted'])[:30]}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "real_images.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nwritten to {args.out / 'real_images.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
