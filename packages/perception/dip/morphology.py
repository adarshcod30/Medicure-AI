"""
Stage 8 — morphological operations.

The reason this stage exists is embossed and debossed text on foil. That text
has almost no *colour* contrast with its background — it is the same aluminium.
What distinguishes it is shading: a raised character catches light on one side
and casts a shadow on the other. Intensity thresholding has nothing to work
with. Morphology, which reasons about local shape rather than absolute value,
does.

The white top-hat (image minus its opening) is the specific tool. Opening with a
structuring element larger than the strokes removes anything stroke-sized and
leaves the background; subtracting that background leaves the strokes, with the
lighting gradient cancelled out along the way — because it is present in both
terms of the subtraction.
"""

from __future__ import annotations

import cv2
import numpy as np

from .acquire import to_gray


def _kernel(size: int, shape: str = "ellipse") -> np.ndarray:
    shapes = {
        "rect": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross": cv2.MORPH_CROSS,
    }
    size = max(1, size) | 1
    return cv2.getStructuringElement(shapes.get(shape, cv2.MORPH_ELLIPSE), (size, size))


def erode(img: np.ndarray, size: int = 3, iterations: int = 1, shape: str = "ellipse"):
    """Local minimum. Thins bright regions, thickens dark text."""
    return cv2.erode(img, _kernel(size, shape), iterations=iterations)


def dilate(img: np.ndarray, size: int = 3, iterations: int = 1, shape: str = "ellipse"):
    """Local maximum. Thickens bright regions, closes stroke breaks in
    inverted (white-text) images."""
    return cv2.dilate(img, _kernel(size, shape), iterations=iterations)


def opening(img: np.ndarray, size: int = 3, shape: str = "ellipse") -> np.ndarray:
    """Erode then dilate. Deletes bright detail smaller than the kernel while
    leaving larger structures at their original size — which is what makes it a
    background estimator for top-hat."""
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, _kernel(size, shape))


def closing(img: np.ndarray, size: int = 3, shape: str = "ellipse") -> np.ndarray:
    """Dilate then erode. Fills small dark gaps — broken strokes in scratched
    printing, or the gaps in a dotted expiry-date stamp."""
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, _kernel(size, shape))


def top_hat(img: np.ndarray, size: int = 15, shape: str = "ellipse") -> np.ndarray:
    """White top-hat: image - opening(image). Isolates *bright* detail.

    The kernel must be larger than the stroke width but smaller than the
    background features you want removed. At the default 15 px it suits text
    printed at roughly 6-10 pt in a ~2000 px-wide photo.
    """
    return cv2.morphologyEx(to_gray(img), cv2.MORPH_TOPHAT, _kernel(size, shape))


def black_hat(img: np.ndarray, size: int = 15, shape: str = "ellipse") -> np.ndarray:
    """Black top-hat: closing(image) - image. Isolates *dark* detail.

    The counterpart to `top_hat`, for the far more common case of dark ink on
    bright packaging.
    """
    return cv2.morphologyEx(to_gray(img), cv2.MORPH_BLACKHAT, _kernel(size, shape))


def text_polarity(img: np.ndarray) -> str:
    """Guess whether text is dark-on-light or light-on-dark.

    Compares the mean intensity of the extreme deciles against the median. The
    result selects top-hat versus black-hat; getting it wrong yields an almost
    empty response, since the operator would be hunting for detail of the wrong
    sign.
    """
    gray = to_gray(img)
    median = float(np.median(gray))
    dark_mass = float((gray < median - 40).sum())
    light_mass = float((gray > median + 40).sum())
    return "light_on_dark" if light_mass > dark_mass * 1.4 else "dark_on_light"


def enhance_text(img: np.ndarray, *, size: int = 15, polarity: str | None = None):
    """Apply whichever top-hat matches the detected text polarity.

    Returns an image in which strokes are bright on a near-black field, ready
    for a global threshold — the lighting gradient has already been removed by
    the subtraction, so a global method is now adequate.
    """
    if polarity is None:
        polarity = text_polarity(img)
    return top_hat(img, size) if polarity == "light_on_dark" else black_hat(img, size)


def remove_small_components(mask: np.ndarray, min_area: int = 12) -> np.ndarray:
    """Drop connected components below `min_area` pixels.

    Post-binarisation cleanup: isolated specks survive thresholding and become
    stray punctuation in the OCR output, which then corrupts the token stream
    the resolver matches against.

    Implemented as a single lookup over the label image rather than a loop of
    `out[labels == i] = 255`. The obvious per-component version scans the whole
    image once per blob, so it costs O(components x pixels) — and after adaptive
    upscaling there are thousands of components on a 3000px image. It accounted
    for roughly 9 of the 17 seconds a full scan was taking. Building a boolean
    keep-mask indexed by label makes it O(pixels).
    """
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return np.zeros_like(mask)

    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False  # label 0 is background
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def skeletonize(mask: np.ndarray, max_iterations: int = 100) -> np.ndarray:
    """Morphological thinning to a 1-px skeleton.

    Used by the stroke-width estimate in `textdetect.py`, not in the OCR path —
    thinned glyphs read considerably worse than solid ones.
    """
    img = mask.copy()
    skeleton = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    for _ in range(max_iterations):
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(img, opened))
        img = cv2.erode(img, element)
        if cv2.countNonZero(img) == 0:
            break

    return skeleton
