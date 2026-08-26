"""
Stage 9 — binarisation.

Tesseract binarises internally (Otsu), so the reason to do it here is that we
can do it *better* for this specific input, and can produce several competing
binarisations and let OCR confidence pick the winner.

The global-versus-local distinction is the whole story:

  **Otsu** picks one threshold for the entire image by maximising between-class
  variance. It is optimal when the histogram is genuinely bimodal — a flat,
  evenly lit label. Under a lighting gradient the "ink" mode of the bright
  region overlaps the "paper" mode of the dark region, the histogram becomes
  unimodal, and any single threshold must sacrifice one end of the image.

  **Sauvola** computes a threshold per pixel from the local mean and standard
  deviation. Its rule, T = m(1 + k(s/R - 1)), has a property that matters here:
  where local contrast is high (s large) the threshold drops toward the mean and
  text is captured; where the neighbourhood is flat (s small, i.e. background)
  the threshold falls *below* the mean, so uniform areas stay uniform instead of
  being speckled. Niblack, which lacks that term, notoriously fills empty
  background with noise.

Local means and variances are computed with box filters. A box filter is a
separable running sum, so the cost is independent of window size — a naive
per-pixel loop over a 25x25 window would be orders of magnitude slower.
"""

from __future__ import annotations

import cv2
import numpy as np

from .acquire import to_gray
from .config import BinarizeMethod


def _local_stats(gray: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Local mean and standard deviation over a square window.

    Uses Var(X) = E[X^2] - E[X]^2. The clamp at zero guards the small negative
    values that float rounding can produce when the true variance is ~0.
    """
    window = max(3, window) | 1
    f = gray.astype(np.float32)

    mean = cv2.boxFilter(f, cv2.CV_32F, (window, window), normalize=True,
                         borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.boxFilter(f * f, cv2.CV_32F, (window, window), normalize=True,
                            borderType=cv2.BORDER_REFLECT)

    variance = np.maximum(mean_sq - mean * mean, 0.0)
    return mean, np.sqrt(variance)


def otsu(img: np.ndarray, *, invert: bool = True) -> np.ndarray:
    """Global Otsu. `invert=True` yields white text on black, which is what the
    morphology helpers and Tesseract's own default expect."""
    gray = to_gray(img)
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, out = cv2.threshold(gray, 0, 255, flag + cv2.THRESH_OTSU)
    return out


def adaptive_mean(img: np.ndarray, *, block: int = 31, c: int = 10, invert: bool = True):
    """Threshold at (local mean - C). The cheapest local method; noisier than
    Sauvola because it ignores local variance entirely."""
    gray = to_gray(img)
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, flag, block | 1, c)


def adaptive_gaussian(img: np.ndarray, *, block: int = 31, c: int = 10, invert: bool = True):
    """Threshold at (Gaussian-weighted local mean - C). Weighting by distance
    makes it less prone than `adaptive_mean` to a bright pixel at the window's
    edge dragging the whole neighbourhood's threshold."""
    gray = to_gray(img)
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, flag, block | 1, c)


def sauvola(
    img: np.ndarray, *, window: int = 25, k: float = 0.2, r: float = 128.0, invert: bool = True
) -> np.ndarray:
    """Sauvola local thresholding: T = m * (1 + k * (s/R - 1)).

    The default for unevenly-lit packaging. `window` should comfortably exceed
    the stroke width — too small and the window sits entirely inside a thick
    stroke, where local contrast is ~0 and the character hollows out.
    """
    gray = to_gray(img)
    mean, std = _local_stats(gray, window)

    threshold = mean * (1.0 + k * ((std / r) - 1.0))
    mask = gray.astype(np.float32) < threshold if invert else gray.astype(np.float32) >= threshold
    return (mask.astype(np.uint8)) * 255


def niblack(
    img: np.ndarray, *, window: int = 25, k: float = -0.2, invert: bool = True
) -> np.ndarray:
    """Niblack local thresholding: T = m + k * s.

    Included for comparison and because it genuinely wins on very low-contrast
    embossed text, where Sauvola's variance term suppresses the faint strokes it
    is meant to find. Its weakness is the mirror image: background regions get
    speckled, so it pairs best with `morphology.remove_small_components`.
    """
    gray = to_gray(img)
    mean, std = _local_stats(gray, window)

    threshold = mean + k * std
    mask = gray.astype(np.float32) < threshold if invert else gray.astype(np.float32) >= threshold
    return (mask.astype(np.uint8)) * 255


def wolf(
    img: np.ndarray, *, window: int = 25, k: float = 0.5, invert: bool = True
) -> np.ndarray:
    """Wolf-Jolion thresholding: T = m + k * (s/S_max - 1) * (m - M_min).

    Normalises the local standard deviation by its global maximum and anchors to
    the global minimum intensity, which makes it more stable than Sauvola when
    contrast varies a lot *between* regions rather than within them — a strip
    half in shadow, for instance.
    """
    gray = to_gray(img)
    mean, std = _local_stats(gray, window)

    s_max = float(std.max()) or 1.0
    m_min = float(gray.min())

    threshold = mean + k * ((std / s_max) - 1.0) * (mean - m_min)
    mask = gray.astype(np.float32) < threshold if invert else gray.astype(np.float32) >= threshold
    return (mask.astype(np.uint8)) * 255


_DISPATCH = {
    "otsu": otsu,
    "adaptive_mean": adaptive_mean,
    "adaptive_gaussian": adaptive_gaussian,
    "sauvola": sauvola,
    "niblack": niblack,
    "wolf": wolf,
}


def binarize(
    img: np.ndarray,
    method: BinarizeMethod = "sauvola",
    *,
    invert: bool = True,
    window: int = 25,
    sauvola_k: float = 0.2,
    niblack_k: float = -0.2,
) -> np.ndarray:
    """Dispatch to a named binarisation method."""
    if method == "sauvola":
        return sauvola(img, window=window, k=sauvola_k, invert=invert)
    if method == "niblack":
        return niblack(img, window=window, k=niblack_k, invert=invert)
    if method == "wolf":
        return wolf(img, window=window, invert=invert)

    fn = _DISPATCH.get(method, sauvola)
    return fn(img, invert=invert)


def binarize_many(
    img: np.ndarray,
    methods: tuple[BinarizeMethod, ...],
    *,
    window: int = 25,
    sauvola_k: float = 0.2,
    niblack_k: float = -0.2,
) -> dict[str, np.ndarray]:
    """Produce several binarisations of the same image.

    Which method wins is genuinely image-dependent and not predictable in
    advance, so rather than guess we generate the candidates and let OCR
    confidence arbitrate in `fuse.py`. This is the "multi-rendition fusion" row
    of the ablation table.
    """
    return {
        m: binarize(img, m, window=window, sauvola_k=sauvola_k, niblack_k=niblack_k)
        for m in methods
    }


def ink_coverage(mask: np.ndarray) -> float:
    """Fraction of the mask that is foreground.

    A cheap plausibility check on a binarisation. Real text lands roughly in
    2-25%. Below that the threshold has erased the text; above it the image has
    been flooded, usually by a shadow admitted as ink. Either way the rendition
    is not worth an OCR pass.
    """
    return float((mask > 0).sum() / mask.size)
