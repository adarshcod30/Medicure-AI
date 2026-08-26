"""
Stage 5 — edge detection.

Feeds boundary detection (`segment.py`) and deskew (`rectify.py`), and is
benchmarked in its own right by `eval/bench_ocr.py`.

The important design decision here is **auto-hysteresis**. Almost every tutorial
writes `cv2.Canny(img, 50, 150)`, which is tuned to one exposure and silently
degrades everywhere else: on a dark photo those thresholds reject real edges, on
a bright one they admit noise. Deriving both thresholds from the image median
makes the operator adapt to whatever exposure it is handed — which is the
difference between a boundary detector that works across a varied photo set and
one that works on the images it was tuned against.
"""

from __future__ import annotations

import cv2
import numpy as np

from .acquire import to_gray
from .config import EdgeMethod


def gradient_magnitude(img: np.ndarray, *, blur: bool = True) -> np.ndarray:
    """Sobel L2 gradient magnitude, in the same units Canny uses internally.

    Kept float and unnormalised on purpose — `auto_canny` needs the true
    magnitudes to pick thresholds, and normalising to 0-255 would throw away the
    scale information that makes those thresholds meaningful.
    """
    gray = to_gray(img)
    if blur:
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def auto_canny(
    img: np.ndarray,
    sigma: float = 0.33,
    *,
    blur: bool = True,
    method: str = "gradient",
    high_percentile: float = 92.0,
    low_ratio: float = 0.4,
) -> np.ndarray:
    """Canny with automatically-chosen hysteresis thresholds.

    Two rules are available, and the difference between them matters a lot here.

    **method="median"** is the widely-repeated rule::

        lower = (1 - sigma) * median(intensity)
        upper = (1 + sigma) * median(intensity)

    It assumes the median intensity sits near mid-grey. Medicine packaging
    breaks that assumption: a strip fills the frame with bright foil and white
    label, pushing the median to ~238, so `upper` computes to ~316. Clamped to
    255 that exceeds every gradient the image contains, the strong-edge set
    comes out empty, hysteresis has no seeds to grow from, and Canny returns
    essentially nothing. Retained here because it is the standard baseline and
    the ablation table should show why it was rejected.

    **method="gradient"** (default) takes the thresholds from the distribution
    of gradient magnitudes instead::

        upper = percentile(|grad| over edge-ish pixels, high_percentile)
        lower = low_ratio * upper

    This asks "how strong are the edges that actually exist in this image?"
    rather than "how bright is this image?", so it is invariant to exposure. The
    thresholds are deliberately *not* clamped to 255 — Sobel L2 magnitudes run
    up to ~1442, and clamping would reintroduce the same failure.

    The leading Gaussian is part of Canny as originally specified; OpenCV's
    implementation omits it, and without it the derivative step turns sensor
    noise into edges.
    """
    gray = to_gray(img)
    if blur:
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

    if method == "median":
        median = float(np.median(gray))
        lower = max(0.0, (1.0 - sigma) * median)
        upper = min(255.0, (1.0 + sigma) * median)
        if upper <= lower:
            upper = lower + 1.0
        return cv2.Canny(gray, lower, upper, L2gradient=True)

    magnitude = gradient_magnitude(gray, blur=False)

    # Ignore the flat majority of the image. Including near-zero gradients would
    # drag any percentile down to noise level, since most pixels of a label are
    # background.
    significant = magnitude[magnitude > 2.0]
    if significant.size < 64:
        return np.zeros(gray.shape, dtype=np.uint8)

    upper = float(np.percentile(significant, high_percentile))
    lower = max(1.0, upper * low_ratio)
    if upper <= lower:
        upper = lower + 1.0

    return cv2.Canny(gray, lower, upper, L2gradient=True)


def sobel(img: np.ndarray, ksize: int = 3, *, magnitude: bool = True) -> np.ndarray:
    """First-derivative operator with built-in smoothing.

    Computed in float64 then normalised: doing it in uint8 would clip every
    negative gradient to zero and lose all dark-to-light transitions.
    """
    gray = to_gray(img)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

    out = np.sqrt(gx**2 + gy**2) if magnitude else np.abs(gx) + np.abs(gy)
    return cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def scharr(img: np.ndarray) -> np.ndarray:
    """Scharr operator — a 3x3 kernel with better rotational symmetry than
    Sobel's. Preferable when gradient *direction* matters, which it does for the
    text-orientation estimate in `rectify.estimate_skew`."""
    gray = to_gray(img)
    gx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    out = np.sqrt(gx**2 + gy**2)
    return cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def laplacian_of_gaussian(img: np.ndarray, sigma: float = 1.4, ksize: int = 5) -> np.ndarray:
    """LoG — blur, then take the second derivative.

    Zero-crossings of the Laplacian mark edges. It is isotropic (no directional
    bias) but doubles noise sensitivity by differentiating twice, so the
    Gaussian is mandatory rather than optional.
    """
    gray = to_gray(img)
    blurred = cv2.GaussianBlur(gray, (ksize | 1, ksize | 1), sigma)
    out = cv2.Laplacian(blurred, cv2.CV_64F, ksize=ksize | 1)
    return cv2.normalize(np.abs(out), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def morphological_gradient(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Dilation minus erosion — the local intensity range.

    Non-linear, so it produces closed contours of near-uniform thickness where
    gradient methods thin out on shallow edges. Useful on embossed foil text,
    where the "edge" is a soft shading ramp rather than a step.
    """
    gray = to_gray(img)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, k)


def detect(img: np.ndarray, method: EdgeMethod = "canny", *, sigma: float = 0.33) -> np.ndarray:
    """Dispatch to the named edge operator."""
    if method == "canny":
        return auto_canny(img, sigma)
    if method == "sobel":
        return sobel(img)
    if method == "scharr":
        return scharr(img)
    if method == "log":
        return laplacian_of_gaussian(img)
    if method == "morph_gradient":
        return morphological_gradient(img)
    return auto_canny(img, sigma)


def edge_density(edges: np.ndarray) -> float:
    """Fraction of pixels that are edges.

    A useful sanity signal: near zero means the image is blank or hopelessly
    blurred; very high means noise is being detected as structure. Both are
    reasons to distrust the boundary quadrilateral.
    """
    return float((edges > 0).sum() / edges.size)
