"""
Stage 4 — contrast and detail enhancement.

CLAHE is the workhorse. Global histogram equalisation fails on medicine
packaging because a strip is not one lighting regime: a bright foil face and a
shadowed fold in the same frame have opposite corrections, and a single global
transfer curve must compromise between them. CLAHE equalises per tile, so each
region gets its own curve.

The *contrast-limited* part matters as much as the adaptive part. In a near-flat
tile — an empty stretch of foil — plain adaptive equalisation stretches sensor
noise across the full range and manufactures texture that looks like text to
both OCR and the boundary detector. Clipping the histogram before equalising
bounds how much any single bin can be amplified, which stops that.
"""

from __future__ import annotations

import cv2
import numpy as np

from .acquire import from_lab, to_gray, to_lab


def clahe(
    img: np.ndarray, *, clip_limit: float = 3.0, tile_grid: int = 8, luminance_only: bool = True
) -> np.ndarray:
    """Contrast-Limited Adaptive Histogram Equalisation.

    On a colour image this runs on the L channel in LAB space. Applying it to
    B, G and R independently would alter their ratios and shift hues — turning
    a white tablet pink is a real failure mode and it also breaks the
    saturation test that glare detection depends on.
    """
    op = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))

    if img.ndim == 2:
        return op.apply(img)

    if luminance_only:
        lab = to_lab(img)
        lab[:, :, 0] = op.apply(lab[:, :, 0])
        return from_lab(lab)

    return cv2.merge([op.apply(c) for c in cv2.split(img)])


def gamma_correct(img: np.ndarray, gamma: float) -> np.ndarray:
    """Power-law transform, via a 256-entry LUT.

    gamma < 1 lifts shadows (recovering text in a fold); gamma > 1 pulls down
    highlights. A LUT is used because the transform is a pure per-pixel function
    of intensity — computing pow() 8 million times would be pointless.
    """
    if gamma <= 0:
        return img
    inv = 1.0 / gamma
    lut = np.array([((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)


def stretch_contrast(img: np.ndarray, *, low_pct: float = 1.0, high_pct: float = 99.0):
    """Linear contrast stretch between two percentiles.

    Percentiles rather than min/max: a single hot pixel or one clipped specular
    dot would otherwise define the range and leave the stretch doing nothing.
    """
    gray = to_gray(img)
    lo, hi = np.percentile(gray, [low_pct, high_pct])
    if hi - lo < 1e-6:
        return gray
    out = (gray.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


def unsharp_mask(
    img: np.ndarray, *, amount: float = 1.0, radius: int = 5, threshold: int = 0
) -> np.ndarray:
    """Sharpen by subtracting a blurred copy.

    sharp = original + amount * (original - blurred). The difference term is a
    high-pass, so this boosts edges. `threshold` suppresses the boost where the
    local difference is small, which keeps flat noisy areas from being sharpened
    into speckle — the usual failure of naive sharpening.
    """
    radius = radius | 1
    blurred = cv2.GaussianBlur(img, (radius, radius), 0)

    if threshold > 0:
        diff = cv2.absdiff(img, blurred)
        mask = (to_gray(diff) if diff.ndim == 3 else diff) >= threshold
        sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
        if img.ndim == 3:
            mask = np.repeat(mask[:, :, None], 3, axis=2)
        return np.where(mask, sharpened, img).astype(np.uint8)

    return cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)


def apply(
    img: np.ndarray,
    *,
    use_clahe: bool = True,
    clip_limit: float = 3.0,
    tile_grid: int = 8,
    gamma: float | None = None,
    unsharp: bool = True,
    unsharp_amount: float = 1.0,
    unsharp_radius: int = 5,
) -> np.ndarray:
    """Run the enhancement chain in the order that composes correctly.

    Gamma first (fix exposure), then CLAHE (fix local contrast), then unsharp
    (fix acutance). Reversing CLAHE and unsharp would have CLAHE amplify the
    halos that sharpening introduces.
    """
    out = img
    if gamma is not None:
        out = gamma_correct(out, gamma)
    if use_clahe:
        out = clahe(out, clip_limit=clip_limit, tile_grid=tile_grid)
    if unsharp:
        out = unsharp_mask(out, amount=unsharp_amount, radius=unsharp_radius, threshold=3)
    return out
