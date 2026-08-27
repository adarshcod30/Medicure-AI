"""
Adaptive upscaling — matching text size to what the OCR engine expects.

This is the single largest source of error on real medicine packaging, and it
is not a preprocessing subtlety. Tesseract's LSTM models are trained around a
capital-letter height of roughly 30 px. Measured across 28 real images, the
median glyph height was **6 to 11 px** — every one of them well under half of
what the recogniser wants.

The effect is not gradual. On one product shot whose composition panel is
perfectly legible to a person, the same pipeline at different scales gave:

    1x   "Amoxycliin Thydrates ... Aaciic Ae Bacllus"
    2x   "Amoxycilin Trihydrate ... Lactic Acid Bacillus 60 million spores"
    4x   "Amoxycillin Trihydrate ... 500 mg Lactic Acid Bacillus"

The drug name is unrecoverable at 1x and exact at 4x. No amount of denoising,
binarisation or contrast work fixes it, because the information is present and
simply too small for the model to resolve. And nothing downstream can recover a
token that OCR never emitted — character n-gram matching degrades gracefully
against a *garbled* name, but has nothing to work with against an absent one.

Upscaling adds no information. It resamples what is there onto a grid the
recogniser can actually use, which is a different and legitimate thing.

The scale is measured rather than fixed, because a fixed multiplier is wrong in
both directions: too small for a 500 px thumbnail, and a needless 4x cost on a
4000 px close-up that is already well resolved.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .acquire import to_gray
from .binarize import sauvola

TARGET_GLYPH_HEIGHT = 30.0
"""Capital-letter height Tesseract's LSTM models are trained around."""

MAX_SCALE = 4.0
"""Cap on the multiplier. Beyond 4x the gain flattens while cost grows with the
square, and interpolation artefacts start to blur strokes together."""

MAX_OUTPUT_PIXELS = 12_000_000
"""Roughly 3500x3500.

Chosen from measured cost. OCR time scales with pixel count: at a 16 MP cap,
upscaling the whole fan-out took a single image from 4.2 to 56.6 seconds. The
fix was to stop multiplying the fan-out rather than to cut the resolution
hard — the drug name needs the pixels, the eight other renditions do not."""

MIN_COMPONENTS = 12
"""Below this many plausible glyphs the height estimate is not trustworthy, and
no scaling is applied. Guessing a 5x upscale from four blobs is how a texture
gets magnified into something OCR reads as text."""


@dataclass
class ScaleEstimate:
    glyph_height: float
    scale: float
    components: int
    applied: bool

    def to_dict(self) -> dict:
        return {
            "glyph_height_px": round(self.glyph_height, 1),
            "scale": round(self.scale, 2),
            "glyph_components": self.components,
            "applied": self.applied,
        }


def estimate_glyph_height(gray: np.ndarray) -> tuple[float, int]:
    """Median height of connected components that look like characters.

    Returns (median height in px, number of components counted).

    The geometric filters matter more than the statistic. Without them the
    median is dominated by noise specks and by large packaging graphics, and
    the resulting scale factor is meaningless:

      * height between 4 px and a quarter of the image — a component taller
        than that is a border or a logo, not a glyph
      * aspect ratio 0.08 to 3.0 — excludes rule lines and tall thin artefacts
      * fill ratio above 0.12 — a real glyph occupies a reasonable share of its
        bounding box, whereas a sparse speckle cluster does not

    The median rather than the mean, because packaging mixes a large brand name
    with small print and the mean sits between the two — describing neither. The
    median lands on the body text, which is where the composition is printed and
    what actually needs resolving.
    """
    mask = sauvola(gray, window=25, k=0.2, invert=True)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    max_height = gray.shape[0] * 0.25
    heights: list[int] = []

    for i in range(1, count):
        width = int(stats[i, cv2.CC_STAT_WIDTH])
        height = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])

        if height < 4 or height > max_height or width == 0:
            continue
        aspect = width / height
        if not (0.08 <= aspect <= 3.0):
            continue
        if area < 0.12 * width * height:
            continue

        heights.append(height)

    if len(heights) < MIN_COMPONENTS:
        return 0.0, len(heights)

    return float(np.median(heights)), len(heights)


def estimate(gray: np.ndarray, *, target: float = TARGET_GLYPH_HEIGHT) -> ScaleEstimate:
    """Decide how much to upscale so glyphs reach `target` height."""
    glyph_height, components = estimate_glyph_height(gray)

    if glyph_height <= 0:
        return ScaleEstimate(0.0, 1.0, components, applied=False)

    scale = target / glyph_height

    height, width = gray.shape[:2]
    pixel_cap = (MAX_OUTPUT_PIXELS / (height * width)) ** 0.5
    scale = min(scale, MAX_SCALE, pixel_cap)

    # Below ~1.2x the resampling is not worth an extra OCR pass.
    if scale < 1.2:
        return ScaleEstimate(glyph_height, 1.0, components, applied=False)

    return ScaleEstimate(glyph_height, scale, components, applied=True)


def upscale(image: np.ndarray, scale: float) -> np.ndarray:
    """Resample by `scale` using cubic interpolation.

    Cubic rather than Lanczos: Lanczos is sharper but rings at high-contrast
    edges, and a ring alongside a character stroke is exactly the artefact a
    binariser turns into an extra mark.
    """
    if scale <= 1.0:
        return image
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def apply(image: np.ndarray, *, target: float = TARGET_GLYPH_HEIGHT):
    """Estimate and apply. Returns (image, estimate)."""
    estimate_result = estimate(to_gray(image), target=target)
    if not estimate_result.applied:
        return image, estimate_result
    return upscale(image, estimate_result.scale), estimate_result
