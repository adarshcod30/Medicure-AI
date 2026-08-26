"""
The DIP pipeline orchestrator.

Runs the stages in an order chosen so that each one operates on input the
previous has already made valid for it:

  acquire -> denoise -> [measure quality] -> glare inpaint -> boundary
          -> rectify -> deskew -> illumination -> CLAHE -> sharpen
          -> morphology -> binarize -> renditions

Four orderings are load-bearing, and three of them were arrived at by watching
the pipeline fail:

  * **Quality is measured before any restoration.** Otherwise the pipeline
    grades its own output: glare fraction reads 0% after inpainting, and a
    12%-blown-out photo looks pristine. Since quality drives abstention, that
    gets the central decision exactly backwards.

  * **Glare inpainting precedes boundary detection.** A specular highlight is a
    high-contrast blob, so Canny outlines it and `approxPolyDP` will happily
    return that outline as "the packet".

  * **Illumination normalisation comes AFTER boundary detection and
    rectification** — not before, which is where it was first placed. It is a
    high-pass, and the packet/background step edge is low-frequency at its
    kernel size, so running it early flattens packet and background to the same
    mid-grey and erases the boundary along with the lighting gradient. Boundary
    detection then finds nothing but the image border.

  * **Rectification precedes CLAHE.** CLAHE equalises per tile. Applied before
    the warp, its tile grid is laid over the *distorted* image, so after warping
    the tiles are stretched non-uniformly and their boundaries become visible
    seams — which the binariser reads as edges.

The pipeline ends with a **fan-out** rather than a single answer. Which
binarisation wins is genuinely image-dependent, so instead of guessing we emit
several renditions and let OCR confidence arbitrate downstream in `fuse.py`.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import acquire, binarize, denoise, edges, enhance, glare, morphology, rectify, segment
from .config import DipConfig
from .quality import QualityReport, assess


@dataclass
class Rendition:
    """One candidate image to hand to OCR."""

    name: str
    image: np.ndarray
    rotation: int = 0
    binarize_method: str = "none"
    ink_coverage: float = 0.0


@dataclass
class DipResult:
    """Everything the DIP layer produces for one input image."""

    original: np.ndarray
    processed: np.ndarray
    """Best enhanced greyscale, pre-binarisation. This is what gets sent to the
    vision model when transcription fallback fires — a vision model wants
    continuous tone, not a 1-bit mask."""

    renditions: list[Rendition]
    quality: QualityReport
    metrics: dict = field(default_factory=dict)
    stages: dict[str, np.ndarray] = field(default_factory=dict)
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        """JSON-safe summary for the API response and the inspector panel."""
        return {
            **self.quality.to_dict(),
            "dip_stages_applied": self.metrics.get("stages_applied", []),
            "boundary_method": self.metrics.get("boundary_method", "none"),
            "denoise_method": self.metrics.get("denoise_method", "none"),
            "rectified": self.metrics.get("rectified", False),
            "renditions": [r.name for r in self.renditions],
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


def run(image: bytes | np.ndarray, config: DipConfig | None = None) -> DipResult:
    """Execute the DIP pipeline.

    Accepts raw bytes (the upload path) or an array (the benchmark path, which
    decodes once and reuses across every ablation rung).
    """
    config = config or DipConfig.full()
    started = time.perf_counter()

    stages: dict[str, np.ndarray] = {}
    metrics: dict = {"stages_applied": []}

    def record(name: str, img: np.ndarray) -> None:
        metrics["stages_applied"].append(name)
        if config.dump_stages:
            stages[name] = img.copy()

    # --- 1. acquire -------------------------------------------------------
    img = acquire.decode(image, auto_orient=config.auto_orient) if isinstance(image, bytes) \
        else image.copy()
    original = img.copy()

    img, scale = acquire.limit_resolution(img, config.max_dimension)
    metrics["resize_scale"] = scale
    record("acquire", img)

    # --- 2. denoise -------------------------------------------------------
    if config.denoise:
        img, used = denoise.apply(img, config.denoise_method)
        metrics["denoise_method"] = used
        record(f"denoise:{used}", img)
    else:
        metrics["denoise_method"] = "none"

    # --- quality is measured HERE, before any restoration ----------------
    # This ordering is not incidental. Quality describes *what the camera
    # captured*, and it is what drives the routing and abstention decision.
    # Measuring after restoration would report the pipeline's own output back to
    # itself: glare fraction reads 0% once the highlights have been inpainted,
    # and RMS contrast is meaningless after illumination normalisation has
    # deliberately flattened the histogram to a fixed mean. Both would make a
    # badly-degraded photo look pristine, which is precisely backwards for a
    # system whose main claim is that it knows when to decline.
    quality = assess(acquire.ensure_bgr(img))

    # --- 3. specular glare removal ---------------------------------------
    # Before boundary detection, because a blown highlight is a high-contrast
    # blob that Canny outlines happily and `approxPolyDP` will gladly return as
    # "the packet".
    #
    # Illumination normalisation deliberately does NOT run here — see stage 6.
    if config.remove_glare:
        repaired, mask = glare.inpaint_glare(
            acquire.ensure_bgr(img),
            method=config.glare_inpaint_method,
            v_threshold=config.glare_v_threshold,
            s_threshold=config.glare_s_threshold,
        )
        img = repaired
        metrics["glare_inpainted"] = bool(mask is not None and mask.any())
        if config.dump_stages and mask is not None:
            stages["glare_mask"] = mask
        record("glare_inpaint", img)

    # --- 4. boundary detection -------------------------------------------
    quad = None
    if config.detect_boundary:
        quad, boundary_method = segment.find_packet_quad(
            acquire.ensure_bgr(img), canny_sigma=config.canny_sigma
        )
        metrics["boundary_method"] = boundary_method
        if config.dump_stages:
            stages["edges"] = edges.detect(img, config.edge_method, sigma=config.canny_sigma)
            if quad is not None:
                overlay = acquire.ensure_bgr(img).copy()
                cv2.polylines(overlay, [quad.astype(np.int32)], True, (0, 255, 0), 3)
                stages["boundary"] = overlay
        metrics["stages_applied"].append(f"boundary:{boundary_method}")
    else:
        metrics["boundary_method"] = "disabled"

    # --- 5. rectify + deskew ---------------------------------------------
    if config.rectify or config.deskew:
        img, geom_metrics = rectify.apply(
            img,
            quad,
            do_rectify=config.rectify,
            do_deskew=config.deskew,
            max_skew_deg=config.max_skew_correction_deg,
        )
        metrics.update(geom_metrics)
        record("rectify+deskew", img)

    # --- 6. illumination normalisation -----------------------------------
    # Deliberately *after* boundary detection and rectification.
    #
    # Illumination normalisation is a high-pass: `divide_illumination` divides
    # by a heavily-blurred copy, and inside a large uniform region that blurred
    # copy simply is the region, so the quotient is 1.0. Run before boundary
    # detection it flattens packet and background to the same mid-grey and
    # erases the very step edge the detector is looking for — the packet
    # boundary is low-frequency at that kernel size, so it goes out with the
    # lighting gradient. Running it here is also cheaper and more accurate: by
    # this point the packet has been cropped and rectified to fill the frame, so
    # the only low-frequency content left genuinely is illumination.
    if config.normalize_illumination:
        if config.illum_method == "divide":
            img = glare.divide_illumination(img)
        elif config.illum_method == "retinex":
            img = glare.multi_scale_retinex(img)
        elif config.illum_method == "homomorphic":
            from .frequency import homomorphic_filter

            img = homomorphic_filter(img)
        metrics["illumination_method"] = config.illum_method
        record(f"illumination:{config.illum_method}", img)

    # --- 7. contrast ------------------------------------------------------
    if config.clahe or config.gamma is not None or config.unsharp:
        img = enhance.apply(
            img,
            use_clahe=config.clahe,
            clip_limit=config.clahe_clip_limit,
            tile_grid=config.clahe_tile_grid,
            gamma=config.gamma,
            unsharp=config.unsharp,
            unsharp_amount=config.unsharp_amount,
            unsharp_radius=config.unsharp_radius,
        )
        record("enhance", img)

    # --- 8. periodic noise ------------------------------------------------
    if config.notch_filter:
        from .frequency import notch_filter

        img = notch_filter(img)
        record("notch", img)

    gray = acquire.to_gray(img)

    # Fold in the geometry measured downstream, and record the sharpness of the
    # restored image alongside the captured sharpness. The pair is informative:
    # a large gap means the pipeline did real work, and a near-zero one on a
    # blurred input means nothing could be recovered and abstention stands.
    quality.skew_deg = metrics.get("skew_deg", 0.0)
    metrics["blur_variance_captured"] = round(quality.blur_variance, 2)
    metrics["blur_variance_restored"] = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)

    # --- 9. morphological text isolation ---------------------------------
    morph = None
    if config.tophat:
        polarity = morphology.text_polarity(gray)
        morph = morphology.enhance_text(gray, size=config.tophat_kernel, polarity=polarity)
        metrics["text_polarity"] = polarity
        record(f"tophat:{polarity}", morph)

    processed = gray

    # --- 10. binarisation fan-out -----------------------------------------
    renditions: list[Rendition] = []

    if config.binarize_methods:
        sources: list[tuple[str, np.ndarray]] = [("gray", gray)]
        if morph is not None:
            sources.append(("tophat", morph))

        for source_name, source in sources:
            masks = binarize.binarize_many(
                source,
                config.binarize_methods,
                window=config.sauvola_window,
                sauvola_k=config.sauvola_k,
                niblack_k=config.niblack_k,
            )
            for method, mask in masks.items():
                coverage = binarize.ink_coverage(mask)
                # Reject implausible binarisations before spending an OCR pass
                # on them: <1.5% means the text was thresholded away, >45% means
                # a shadow was admitted as ink and the image is flooded.
                if not (0.015 <= coverage <= 0.45):
                    continue

                cleaned = morphology.remove_small_components(mask, min_area=10)
                for rotation in config.rotations:
                    renditions.append(
                        Rendition(
                            name=f"{source_name}:{method}:rot{rotation}",
                            image=rectify.rotate(cleaned, -rotation) if rotation else cleaned,
                            rotation=rotation,
                            binarize_method=method,
                            ink_coverage=coverage,
                        )
                    )

        # Prefer higher-information renditions when the cap bites: ink coverage
        # nearest ~8% is the most text-like.
        renditions.sort(key=lambda r: abs(r.ink_coverage - 0.08))
        renditions = renditions[: config.max_renditions]

    # Always include the unbinarised greyscale. Tesseract's own internal
    # thresholding sometimes beats all of ours, and on the `raw` ablation rung
    # this is the only rendition there is.
    renditions.append(
        Rendition(name="gray:none:rot0", image=acquire.upscale_if_tiny(gray), rotation=0)
    )

    if config.dump_stages:
        for r in renditions:
            stages[f"rendition_{r.name}"] = r.image

    return DipResult(
        original=original,
        processed=processed,
        renditions=renditions,
        quality=quality,
        metrics=metrics,
        stages=stages,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
    )


def select_config(quality: QualityReport) -> tuple[DipConfig, str]:
    """Choose a DIP preset from measured image quality.

    There is no single best pipeline, and this was measured rather than assumed.
    On clean input the full pipeline is actively harmful — token F1 of 0.70
    against 0.93 for no processing at all — because there is nothing to restore
    and the rendition fan-out only contributes noise. On degraded input the
    ranking inverts: 0.46 against 0.21. Running one fixed configuration means
    accepting the wrong half of that trade on every image.

    Note what quality does *not* measure: perspective. A perfectly exposed,
    perfectly focused photo taken at 30 degrees scores "good" while still badly
    needing rectification. That is why the "good" branch selects `light` rather
    than `raw` — geometry correction always runs, and only the photometric
    fan-out is skipped.

    These thresholds are fitted to synthetic images and should be refitted
    against the real photo set; `eval/bench_ocr.py` produces exactly the table
    needed to do that.
    """
    if quality.verdict == "good":
        return DipConfig.light(), "light"
    if quality.verdict == "degraded":
        return DipConfig.fast(), "fast"
    return DipConfig.full(), "full"


def run_auto(image: bytes | np.ndarray, *, dump_stages: bool = False) -> DipResult:
    """Assess quality cheaply, then run the preset that suits the image.

    The probe costs one decode plus three cheap measurements, which is far less
    than the fan-out it avoids on a clean photo.
    """
    decoded = acquire.decode(image) if isinstance(image, bytes) else image
    bounded, _ = acquire.limit_resolution(decoded, DipConfig().max_dimension)

    # Denoise before probing, matching what `run` does before it measures.
    # Probing the raw decode instead gives a different verdict than the one the
    # response ends up reporting — sensor noise inflates the Laplacian variance,
    # so a blurred-but-noisy photo probes as merely "degraded" and then reports
    # "poor" after denoising. The routing decision and the number shown to the
    # user have to come from the same image.
    probed, _ = denoise.apply(bounded, "auto")

    probe = assess(acquire.ensure_bgr(probed))
    config, name = select_config(probe)
    if dump_stages:
        config = config.with_(dump_stages=True)

    result = run(bounded, config)
    result.metrics["auto_preset"] = name
    return result


# --- CLI ------------------------------------------------------------------
# `python -m packages.perception.dip.pipeline --image strip.jpg --dump-stages out/`
# is the fastest way to confirm the DIP work is real: it writes every
# intermediate to disk so you can look at the Canny map, the detected quad, the
# glare mask and each binarisation side by side.


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the MediCure DIP pipeline on an image.")
    parser.add_argument("--image", required=True, type=Path, help="input image path")
    parser.add_argument("--dump-stages", type=Path, help="directory to write intermediates to")
    parser.add_argument(
        "--preset",
        default="auto",
        choices=["auto", "raw", "light", "fast", "full", "foil"],
        help="DipConfig preset (auto selects from measured quality)",
    )
    args = parser.parse_args(argv)

    if not args.image.exists():
        print(f"error: {args.image} not found", file=sys.stderr)
        return 1

    if args.preset == "auto":
        result = run_auto(args.image.read_bytes(), dump_stages=bool(args.dump_stages))
    else:
        config = {
            "raw": DipConfig.raw,
            "light": DipConfig.light,
            "fast": DipConfig.fast,
            "full": DipConfig.full,
            "foil": DipConfig.foil,
        }[args.preset]()
        if args.dump_stages:
            config = config.with_(dump_stages=True)
        result = run(args.image.read_bytes(), config)

    print(f"preset          : {args.preset}"
          + (f" -> {result.metrics['auto_preset']}" if "auto_preset" in result.metrics else ""))
    print(f"elapsed         : {result.elapsed_ms:.0f} ms")
    print(f"verdict         : {result.quality.verdict}")
    print(f"blur variance   : {result.quality.blur_variance:.1f}")
    print(f"glare fraction  : {result.quality.glare_fraction:.1%}")
    print(f"rms contrast    : {result.quality.rms_contrast:.1f}")
    print(f"skew            : {result.metrics.get('skew_deg', 0.0)}°")
    print(f"boundary        : {result.metrics.get('boundary_method', 'n/a')}")
    print(f"rectified       : {result.metrics.get('rectified', False)}")
    print(f"renditions      : {len(result.renditions)}")
    for r in result.renditions:
        print(f"  - {r.name}  (ink {r.ink_coverage:.1%})")
    if result.quality.reasons:
        print(f"issues          : {'; '.join(result.quality.reasons)}")
    if result.quality.advice:
        print(f"advice          : {' '.join(result.quality.advice)}")

    if args.dump_stages:
        args.dump_stages.mkdir(parents=True, exist_ok=True)
        for name, img in result.stages.items():
            safe = name.replace(":", "_").replace("/", "_")
            cv2.imwrite(str(args.dump_stages / f"{safe}.png"), img)
        print(f"\nwrote {len(result.stages)} stage images to {args.dump_stages}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
