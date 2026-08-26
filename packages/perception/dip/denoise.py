"""
Stage 2 — noise estimation and removal.

The filter is chosen from a measurement rather than fixed, because the wrong
filter is actively harmful here: a Gaussian blur strong enough to suppress
sensor noise also dissolves the thin strokes of 6-point composition text.

Selection logic (`method="auto"`):

  impulse noise present  -> median          (the only filter that removes it)
  sigma > 12             -> Non-Local Means (heavy, preserves structure, slow)
  sigma > 4              -> bilateral       (edge-preserving, cheap)
  otherwise              -> no filtering    (image is already clean)
"""

from __future__ import annotations

import cv2
import numpy as np

from .acquire import to_gray
from .config import DenoiseMethod

# Immerkaer's noise-estimation mask: a difference-of-Laplacians kernel whose
# response to a smooth image is ~0, so its mean absolute response measures noise.
_IMMERKAER_MASK = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)


def estimate_noise_sigma(img: np.ndarray) -> float:
    """Estimate additive Gaussian noise sigma (Immerkaer, 1996).

    Convolving with a kernel that annihilates locally-linear intensity leaves
    only the noise. The sqrt(pi/2) factor converts mean-absolute-deviation to
    standard deviation for a Gaussian.
    """
    gray = to_gray(img).astype(np.float64)
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0

    response = cv2.filter2D(gray, -1, _IMMERKAER_MASK, borderType=cv2.BORDER_REPLICATE)
    sigma = np.abs(response).sum() * np.sqrt(np.pi / 2.0) / (6.0 * (w - 2) * (h - 2))
    return float(sigma)


def estimate_impulse_fraction(img: np.ndarray) -> float:
    """Fraction of pixels that look like salt-and-pepper noise.

    A pixel counts only if it is saturated (0 or 255) *and* differs sharply from
    its local median. The second condition is what stops legitimately black text
    or a blown-white background from being mistaken for impulse noise.
    """
    gray = to_gray(img)
    saturated = (gray == 0) | (gray == 255)
    if not saturated.any():
        return 0.0

    med = cv2.medianBlur(gray, 3)
    deviates = np.abs(gray.astype(np.int16) - med.astype(np.int16)) > 60
    return float((saturated & deviates).sum() / gray.size)


def select_method(img: np.ndarray) -> DenoiseMethod:
    """Pick a denoising method from measured noise characteristics."""
    if estimate_impulse_fraction(img) > 0.002:
        return "median"

    sigma = estimate_noise_sigma(img)
    if sigma > 12.0:
        return "nlm"
    if sigma > 4.0:
        return "bilateral"
    return "none"


# --- individual filters ---------------------------------------------------


def gaussian(img: np.ndarray, ksize: int = 5, sigma: float = 0.0) -> np.ndarray:
    ksize = ksize | 1  # OpenCV requires an odd kernel
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def median(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Order-statistic filter. The correct and essentially only choice for
    impulse noise — it discards outliers rather than averaging them in."""
    return cv2.medianBlur(img, ksize | 1)


def bilateral(img: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75):
    """Edge-preserving smoothing.

    Weights neighbours by both spatial distance and intensity similarity, so
    averaging happens within a region but not across an edge. That property is
    what lets us suppress noise without eroding character strokes.
    """
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def non_local_means(img: np.ndarray, h: float = 10.0) -> np.ndarray:
    """Non-Local Means.

    Averages pixels weighted by the similarity of their surrounding *patches*,
    anywhere in the image — not just nearby. Printed text is highly repetitive,
    so NLM finds many genuine matches and denoises hard while keeping strokes
    crisp. It is by far the slowest filter here, hence the resolution cap in
    `acquire.limit_resolution` running first.
    """
    if img.ndim == 3:
        return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)
    return cv2.fastNlMeansDenoising(img, None, h, 7, 21)


def apply(img: np.ndarray, method: DenoiseMethod = "auto") -> tuple[np.ndarray, str]:
    """Denoise `img`, returning the result and the method actually used.

    The returned name is recorded in the response so the DIP inspector can show
    which branch ran, and so ablation results stay interpretable.
    """
    if method == "auto":
        method = select_method(img)

    if method == "none":
        return img, "none"
    if method == "gaussian":
        return gaussian(img), "gaussian"
    if method == "median":
        return median(img), "median"
    if method == "bilateral":
        return bilateral(img), "bilateral"
    if method == "nlm":
        return non_local_means(img), "nlm"

    return img, "none"
