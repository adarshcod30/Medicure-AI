"""
Stage 6 — segmentation: find the packet.

Locating the strip's outline and discarding everything else is worth more than
it first appears. Backgrounds in real photos are tables, hands and bedsheets;
their texture contributes edges that pull the deskew estimate off, contributes
intensities that skew Otsu's global threshold, and contributes regions that
Tesseract dutifully tries to read as text.

The primary route is contour-based: edges -> morphological closing -> external
contours -> the largest 4-sided convex polygon. Two fallbacks follow, because a
strip photographed against a similarly-coloured surface may have no closed
contour at all — and returning a wrong quadrilateral is worse than returning
none, since `rectify` would then warp the image on a false premise.
"""

from __future__ import annotations

import cv2
import numpy as np

from .acquire import to_gray
from .edges import auto_canny


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left.

    Uses the standard coordinate-sum/difference trick: (x+y) is smallest at the
    top-left and largest at the bottom-right; (y-x) is smallest at the top-right
    and largest at the bottom-left. Consistent ordering is a hard prerequisite
    for `getPerspectiveTransform` — mismatched correspondences produce a warp
    that mirrors or rotates the result.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]

    d = np.diff(pts, axis=1).ravel()
    ordered[1] = pts[np.argmin(d)]
    ordered[3] = pts[np.argmax(d)]

    return ordered


def _quad_area(quad: np.ndarray) -> float:
    return float(cv2.contourArea(quad.astype(np.float32).reshape(-1, 1, 2)))


_WORKING_WIDTH = 700
"""Boundary detection runs at a reduced scale. See `_coarse_view` for why."""


def _coarse_view(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Produce a text-suppressed, coarse-scale view for boundary detection.

    This is a scale-space argument, and it is the crux of making boundary
    detection work on a printed package. The packet outline and the printed text
    are structures at *very different scales*, but a single Canny pass sees only
    one gradient distribution containing both. Text edges are far more numerous
    and usually far stronger (dark ink on white label) than the packet's own
    boundary (which may be a soft shadow against a table). Any threshold high
    enough to reject noise is therefore set by the text, and the genuine
    boundary falls below it and is discarded.

    Downscaling and blurring removes the fine scale before thresholding, so the
    gradient distribution is dominated by large-scale structure — which is what
    we are actually looking for. Median blur is used rather than Gaussian
    because it removes text strokes outright instead of smearing them into
    low-contrast ghosts that still register as edges.

    Returns the coarse image and the scale factor, so the quad can be mapped
    back to full resolution.
    """
    h, w = img.shape[:2]
    scale = _WORKING_WIDTH / float(max(h, w)) if max(h, w) > _WORKING_WIDTH else 1.0

    small = (
        cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else img.copy()
    )

    gray = to_gray(small)
    gray = cv2.medianBlur(gray, 9)
    gray = cv2.bilateralFilter(gray, 9, 100, 100)
    return gray, scale


def _candidate_edge_maps(coarse: np.ndarray):
    """Yield (name, edge_map) candidates, cheapest and most reliable first.

    Several detectors are tried rather than one, because the packet/background
    boundary can be any of: a strong intensity step (strip on a dark table), a
    weak step (white box on a white sheet), or effectively no step at all with
    only a shadow to go on. No single operator covers all three.
    """
    yield "otsu", _otsu_boundary(coarse)
    yield "canny_p92", auto_canny(coarse, method="gradient", high_percentile=92.0)
    yield "canny_p70", auto_canny(coarse, method="gradient", high_percentile=70.0)
    yield "canny_p50", auto_canny(coarse, method="gradient", high_percentile=50.0)


def _otsu_boundary(coarse: np.ndarray) -> np.ndarray:
    """Region-based boundary: Otsu-threshold, then take the region outline.

    At the coarse scale the histogram really is close to bimodal — packet versus
    background — because the text that would have muddied it has been blurred
    away. That makes Otsu, which fails on the full-resolution image, a strong
    detector here. Whichever of the two classes touches the frame border is
    taken to be background, which is a safe assumption for a photo of an object.
    """
    _, binary = cv2.threshold(coarse, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    border = np.concatenate(
        [binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]]
    )
    if (border > 0).mean() > 0.5:
        binary = cv2.bitwise_not(binary)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    return cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)


def _quad_from_edges(
    edge_map: np.ndarray,
    image_area: float,
    *,
    min_area_fraction: float,
    max_area_fraction: float,
    top_contours: int,
) -> tuple[np.ndarray | None, str]:
    """Extract the best quadrilateral from one edge map."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(edge_map, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed = cv2.dilate(closed, kernel, iterations=1)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, "none"

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:top_contours]

    # Pass 1 — an exact 4-gon from polygon approximation.
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * min_area_fraction:
            break  # sorted descending, so everything after is smaller too

        perimeter = cv2.arcLength(contour, True)
        # epsilon at 2% of perimeter: tight enough to keep true corners, loose
        # enough to absorb the ragged pixels along a torn edge.
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            if area <= image_area * max_area_fraction:
                return order_points(approx.reshape(4, 2)), "contour_quad"

    # Pass 2 — convex hull, then re-approximate. Recovers the case where a tear
    # or an occluding finger adds spurious concave vertices to a real rectangle.
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * min_area_fraction:
            break
        hull = cv2.convexHull(contour)
        approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
        if len(approx) == 4 and cv2.contourArea(approx) <= image_area * max_area_fraction:
            return order_points(approx.reshape(4, 2)), "convex_hull_quad"

    # Pass 3 — minimum-area rotated rectangle around the largest contour. This
    # always produces 4 points, so it is only trustworthy when the rectangle
    # genuinely fits the contour; otherwise we would warp to a box that does not
    # correspond to the packet, which is worse than not warping at all.
    largest = contours[0]
    area = cv2.contourArea(largest)
    if image_area * min_area_fraction <= area:
        box = cv2.boxPoints(cv2.minAreaRect(largest))
        box_area = max(_quad_area(box), 1.0)
        if box_area <= image_area * max_area_fraction and area / box_area > 0.70:
            return order_points(box), "min_area_rect"

    return None, "none"


def find_packet_quad(
    img: np.ndarray,
    *,
    min_area_fraction: float = 0.12,
    max_area_fraction: float = 0.90,
    canny_sigma: float = 0.33,
    top_contours: int = 6,
) -> tuple[np.ndarray | None, str]:
    """Locate the packet's bounding quadrilateral.

    Returns (ordered 4x2 float32 points in *original* image coordinates, method)
    or (None, "none").

    Detection runs on a coarse, text-suppressed view (see `_coarse_view`) and
    tries several edge maps in order of reliability, accepting the first that
    yields a plausible quadrilateral.

    `min_area_fraction` rejects small rectangles *inside* the packet — a logo
    panel, a single blister pocket — that would otherwise be mistaken for the
    packet. `max_area_fraction` rejects a quad that is essentially the whole
    frame, which means the detector locked onto the image border; rectifying
    against that is a no-op at best and a spurious warp at worst.

    Returning None is a legitimate and useful outcome. `rectify.apply` then
    skips the perspective warp rather than warping on a false premise, which
    would corrupt an image that was merely un-segmentable rather than distorted.
    """
    coarse, scale = _coarse_view(img)
    coarse_area = float(coarse.shape[0] * coarse.shape[1])

    # All detectors are run and the results *scored*, rather than taking the
    # first that succeeds. The detectors have complementary failure modes —
    # region-based Otsu is robust on a cluttered background where Canny finds
    # edges everywhere, and Canny localises corners more precisely when there is
    # a clean intensity step — so which one is best is image-dependent and not
    # knowable in advance. First-match ordering would systematically hand the
    # decision to whichever happened to be listed first.
    best: tuple[float, np.ndarray, str] | None = None

    for name, edge_map in _candidate_edge_maps(coarse):
        quad, method = _quad_from_edges(
            edge_map,
            coarse_area,
            min_area_fraction=min_area_fraction,
            max_area_fraction=max_area_fraction,
            top_contours=top_contours,
        )
        if quad is None or _hugs_frame(quad, coarse.shape[:2]):
            continue

        score = _score_quad(quad, method, coarse_area)
        if best is None or score > best[0]:
            best = (score, quad, f"{name}:{method}")

    if best is None:
        return None, "none"

    _, quad, method = best
    if scale < 1.0:
        quad = quad / scale  # map back to full resolution
    return quad.astype(np.float32), method


def _hugs_frame(quad: np.ndarray, shape: tuple[int, int], *, margin_fraction: float = 0.02):
    """True if the quad is really the image frame rather than the packet.

    Counts corners lying on *any* frame edge and rejects at three or more.

    Two weaker tests were tried first and both let this through. An area test
    alone fails because a frame-spanning trapezoid can measure 0.87 of the image
    and slip under `max_area_fraction`. Requiring every corner to be near a
    frame *corner* also fails: the actual bad quad was
    `[(117,0), (858,0), (885,497), (64,497)]` on a 500x900 image — every corner
    sits on the top or bottom edge, but none is near a corner, so the test
    passed it and the pipeline "rectified" the background instead of the packet.

    Rejecting a genuine packet that fills the frame is a harmless false
    positive: if it already fills the frame there is no perspective to correct,
    and skipping the warp leaves the image untouched.
    """
    h, w = shape
    margin = max(3.0, min(h, w) * margin_fraction)

    on_edge = sum(
        1
        for x, y in quad
        if x <= margin or x >= w - margin or y <= margin or y >= h - margin
    )
    return on_edge >= 3


_METHOD_RANK = {"contour_quad": 3.0, "convex_hull_quad": 2.0, "min_area_rect": 1.0}


def _score_quad(quad: np.ndarray, method: str, image_area: float) -> float:
    """Score a candidate quadrilateral. Higher is better.

    Three terms:

      * **Provenance.** A quad that came out of `approxPolyDP` as an exact
        4-gon is direct evidence of a quadrilateral in the image.
        `min_area_rect` merely fits a box around whatever was there and always
        succeeds, so it is the weakest evidence and ranks last.

      * **Coverage.** Larger is better, because the common failure is locking
        onto something *inside* the packet — a logo panel, one blister pocket.

      * **Right-angledness.** A packet photographed from a plausible angle
        still has roughly perpendicular corners. A quad with a 30-degree corner
        is a spurious shape, and warping to it would shear the image badly.
    """
    rank = _METHOD_RANK.get(method, 0.5)
    coverage = min(_quad_area(quad) / image_area, 1.0)

    # Mean absolute deviation of the four interior angles from 90 degrees.
    deviation = 0.0
    for i in range(4):
        prev_pt, this_pt, next_pt = quad[i - 1], quad[i], quad[(i + 1) % 4]
        v1, v2 = prev_pt - this_pt, next_pt - this_pt
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0
        cosine = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        deviation += abs(np.degrees(np.arccos(cosine)) - 90.0)
    deviation /= 4.0

    squareness = max(0.0, 1.0 - deviation / 45.0)

    return rank + coverage + squareness


def otsu_mask(img: np.ndarray, *, invert: bool = False) -> np.ndarray:
    """Global Otsu threshold — maximises between-class variance.

    Optimal when the histogram is genuinely bimodal (flat, evenly-lit label).
    Under a lighting gradient the two modes smear together and it fails, which
    is precisely why `binarize.py` also offers Sauvola.
    """
    gray = to_gray(img)
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, mask = cv2.threshold(gray, 0, 255, flag + cv2.THRESH_OTSU)
    return mask


def grabcut_foreground(img: np.ndarray, *, margin: float = 0.06, iterations: int = 3):
    """Separate packet from background with GrabCut.

    Models foreground and background as Gaussian mixtures and cuts the graph
    between them. Initialised from a rectangle inset by `margin`, on the
    assumption the user framed the packet roughly centred. Slower than contour
    detection, so it is a fallback rather than the default route.
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (int(w * margin), int(h * margin), int(w * (1 - 2 * margin)), int(h * (1 - 2 * margin)))

    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img, mask, rect, bgd, fgd, iterations, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return np.full((h, w), 255, np.uint8)

    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


def watershed_blisters(img: np.ndarray) -> np.ndarray:
    """Separate touching blister pockets with the watershed transform.

    Treats intensity as terrain and floods from markers, so adjacent pockets
    that a plain threshold would merge into one blob get split at their ridge.
    Used for pocket counting (pack-size inference), not for text extraction.
    """
    gray = to_gray(img)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opened, kernel, iterations=3)

    # Distance transform: pixels far from any background pixel are confidently
    # interior, and give one marker per pocket even when pockets touch.
    dist = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    unknown = cv2.subtract(sure_bg, sure_fg)

    n_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # reserve 0 for the "unknown" band
    markers[unknown == 255] = 0

    markers = cv2.watershed(img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
                            markers)
    return markers


def crop_to_mask(img: np.ndarray, mask: np.ndarray, *, pad: int = 8) -> np.ndarray:
    """Crop to the mask's bounding box, with padding."""
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return img
    h, w = img.shape[:2]
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad)
    return img[y0:y1, x0:x1]
