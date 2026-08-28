#!/usr/bin/env python3
"""
Benchmark on real phone photographs of Indian medicine packaging.

Twelve images, hand-labelled from the packaging itself, carrying the defects
that actually occur: one upside down, four rotated 90 degrees, two badly
crumpled, two bilingual, one held in a hand on a curved tin.

Small, and worth more than the 1,200-query synthetic set. It exposed three bugs
the synthetic benchmark could not:

  * 180-degree rotation was never tried. The rendition fan-out used
    (0, 90, 270), so an upside-down strip was unreadable by construction.
  * The adaptive router disabled rotation for "good quality" images — and four
    of the five rotated photos scored "good", because quality measures
    exposure, focus and glare and says nothing about orientation.
  * OCR reliably returned packaging boilerplate ("store in a cool dry place",
    "keep out of reach of children") rather than the composition, because it is
    set in a dense even block that reads better than a stylised brand name.

Scored on *ingredient recall*: does the predicted composition contain the
ingredients printed on the pack? Exact signature matching would be the wrong
measure here, because several of these are products the index cannot represent
at all (an Ayurvedic plaster, an ORS sachet, a VapoRub tin).

    python -m eval.bench_photos
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from packages.perception import boilerplate, tesseract_engine as te  # noqa: E402
from packages.perception import vision_transcribe  # noqa: E402
from packages.perception.dip import acquire  # noqa: E402
from packages.perception.dip.pipeline import run_auto  # noqa: E402
from packages.resolver.calibrate import load_or_default  # noqa: E402
from packages.resolver.index import get_index  # noqa: E402


def _ingredient_hit(predicted: str, expect: list) -> bool:
    """Does the predicted composition name any expected active ingredient?"""
    low = (predicted or "").lower()
    return any(word.lower() in low for word in expect)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=REPO_ROOT / "data" / "raw" / "photos")
    parser.add_argument("--labels", type=Path,
                        default=REPO_ROOT / "eval" / "datasets" / "photo_labels.json")
    parser.add_argument("--max-dimension", type=int, default=1600)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "eval" / "results")
    parser.add_argument("--vision", action="store_true",
                        help="also run Bedrock vision transcription (costs tokens)")
    parser.add_argument("--text-detection", dest="text_detection",
                        action="store_true", default=None,
                        help="force the MSER text-crop rendition ON")
    parser.add_argument("--no-text-detection", dest="text_detection",
                        action="store_false",
                        help="force the MSER text-crop rendition OFF")
    parser.add_argument(
        "--orchestrator",
        action="store_true",
        help="measure the ORCHESTRATOR path instead of the raw pipeline — "
             "this is what production actually runs",
    )
    args = parser.parse_args(argv)

    if not args.labels.exists():
        print(f"error: {args.labels} not found", file=sys.stderr)
        return 1

    spec = json.loads(args.labels.read_text())
    entries = spec["photos"] + spec.get("web", [])
    index = get_index()
    calibrator = load_or_default()
    stopwords = boilerplate.build_stopwords(index.discriminative_vocabulary())

    transcriber = None
    if args.vision:
        from apps.api.config import get_settings
        from packages.reasoning.bedrock import BedrockClient

        cfg = get_settings()
        transcriber = vision_transcribe.VisionTranscriber(
            BedrockClient(
                region=cfg.aws_region, model_id=cfg.bedrock_model_id,
                fast_model_id=cfg.bedrock_fast_model_id,
                max_tokens=cfg.bedrock_max_tokens, temperature=cfg.bedrock_temperature,
                access_key_id=cfg.aws_access_key_id,
                secret_access_key=cfg.aws_secret_access_key,
            )
        )
        print("vision transcription ENABLED")

    orchestrator = None
    if args.orchestrator:
        from apps.api.config import get_settings
        from apps.api.deps import AppState
        state = AppState()
        state.startup(get_settings())
        orchestrator = state.orchestrator
        print(f"measuring the ORCHESTRATOR path "
              f"(vision {'on' if orchestrator.transcriber else 'off'})")

    rows = []
    started = time.perf_counter()

    for entry in entries:
        path = args.dir.parent / entry.get("dir", "photos") / entry["file"]
        if not path.exists():
            continue

        # --orchestrator measures the path production actually serves, which
        # includes the quality gate, the ingredient-narrowing retry and the
        # vision rescue. The raw path below sees none of those, and that gap
        # hid four real bugs: it reported an unchanged 13/28 while the
        # orchestrator went from 10/28 to 21/28 on the same images. A benchmark
        # that cannot see the code under test is worse than no benchmark,
        # because it reads as reassurance.
        if args.orchestrator:
            result = orchestrator.analyse_image(path.read_bytes(), explain=False)
            ident = result.identification
            predicted = ident.composition or ""
            expected = [e.lower() for e in entry["expect"]]
            low = predicted.lower()
            top5 = " ".join(
                str(c.get("composition", "")) for c in ident.candidates
            ).lower()
            rows.append({
                "file": entry["file"],
                "source": entry.get("source", "phone_capture"),
                "brand": entry.get("brand"),
                "defects": entry.get("defects", []),
                "expect": expected,
                "predicted": predicted or None,
                "hit": bool(expected) and any(e in low for e in expected),
                "hit_top5": bool(expected) and any(e in top5 for e in expected),
                "representable": bool(expected),
                "status": ident.status,
                "probability": round(ident.probability, 3),
                "quality": (result.image_quality or {}).get("verdict", "n/a"),
                "orientation": 0,
                "ocr_confidence": round((result.ocr or {}).get("mean_confidence", 0.0), 1),
                "tokens_raw": len((result.ocr or {}).get("tokens", []) or []),
                "tokens_kept": result.stages.get("narrowed_tokens", 0),
                "vision": bool(result.vision),
                "tokens": ((result.ocr or {}).get("tokens") or [])[:12],
            })
            continue

        image = acquire.decode(path.read_bytes())
        image, _ = acquire.limit_resolution(image, args.max_dimension)

        dip = run_auto(image, text_detection=args.text_detection)
        ocr = te.read_renditions(dip.renditions)
        raw_tokens = ocr.consensus_tokens or ocr.tokens
        tokens = boilerplate.filter_tokens(raw_tokens, stopwords)

        used_vision = False
        if transcriber and dip.quality.use_vision_fallback:
            t = transcriber.transcribe(dip.processed)
            if t.available and t.tokens:
                vt = boilerplate.filter_tokens(t.tokens, stopwords)
                tokens, _attr = vision_transcribe.merge_tokens(tokens, vt)
                used_vision = True

        matches = (
            index.search_compositions_from_tokens(tokens, strengths=ocr.strengths, top_k=5)
            if tokens else []
        )
        status, probability = calibrator.decide(matches, " ".join(tokens))

        expected = [e.lower() for e in entry["expect"]]
        predicted = (matches[0].label.lower() if matches else "")
        top5 = " ".join(m.label.lower() for m in matches)

        # Ingredient recall, not exact signature. Several of these products
        # (an Ayurvedic plaster, an ORS sachet, a VapoRub tin) have no
        # representation in a tablet index at all.
        hit = bool(expected) and any(e in predicted for e in expected)
        hit5 = bool(expected) and any(e in top5 for e in expected)
        representable = bool(expected)

        rows.append({
            "file": entry["file"], "source": entry.get("source", "phone_capture"),
            "brand": entry["brand"], "defects": entry["defects"],
            "expect": expected, "predicted": matches[0].label if matches else None,
            "hit": hit, "hit_top5": hit5, "representable": representable,
            "status": status, "probability": round(probability, 3),
            "quality": dip.quality.verdict,
            "orientation": dip.metrics.get("orientation", {}).get("angle", 0),
            "ocr_confidence": round(ocr.mean_confidence, 1),
            "tokens_raw": len(raw_tokens), "tokens_kept": len(tokens),
            "vision": used_vision,
            "tokens": tokens[:12],
        })

    elapsed = time.perf_counter() - started

    # Reported per source, never pooled. Studio product photography is a far
    # easier distribution than a phone photo of a torn strip, and averaging the
    # two would report a headline number that describes neither.
    print("=" * 82)
    print("REAL IMAGE BENCHMARK")
    print("=" * 82)

    for source in ("phone_capture", "web_product_shot"):
        group = [r for r in rows if r["source"] == source]
        if not group:
            continue
        scored = [r for r in group if r["representable"]]
        answered = [r for r in group if r["status"] in {"confident", "ambiguous"}]
        wrong_confident = [r for r in group if r["status"] == "confident" and not r["hit"]]

        print(f"\n-- {source} " + "-" * (78 - len(source)))
        print(f"  images                     {len(group)} ({len(scored)} representable)")
        print(f"  ingredient hit @1          {sum(r['hit'] for r in scored)}/{len(scored)}"
              f"  ({sum(r['hit'] for r in scored) / max(len(scored), 1):.0%})")
        print(f"  ingredient hit @5          {sum(r['hit_top5'] for r in scored)}/{len(scored)}")
        print(f"  answered                   {len(answered)}/{len(group)}")
        print(f"  SILENT FAILURE             {len(wrong_confident)}/{len(group)}")
        print(f"  orientation corrected      {sum(1 for r in group if r['orientation'])}/{len(group)}")
        print(f"  vision transcription used  {sum(1 for r in group if r.get('vision'))}/{len(group)}")
        print(f"  boilerplate removed        "
              f"{sum(r['tokens_raw'] - r['tokens_kept'] for r in group)} tokens")
        for r in group:
            mark = ("HIT " if r["hit"] else "top5" if r["hit_top5"]
                    else "n/a " if not r["representable"] else "MISS")
            rot = f"rot{r['orientation']}" if r["orientation"] else "----"
            print(f"    [{mark}] {r['file'][:12]:12s} {rot:5s} {r['status'][:10]:10s} "
                  f"P={r['probability']:.2f} {str(r['predicted'])[:32]:32s} <- {r['brand'][:24]}")

    print(f"\n  seconds per image          {elapsed / max(len(rows), 1):.1f}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "photos.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nwritten to {args.out / 'photos.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
