"""
Stage 10 — text region detection.

Not every part of a medicine strip is text. Barcodes, batch-code dot matrices,
manufacturer logos and blister pockets all survive binarisation and all get
handed to Tesseract, which returns confident nonsense for them. That nonsense
then flows into the token stream the resolver matches against, where it does
real damage — a stray "H1N" token is exactly the kind of thing that pulls a
fuzzy match toward the wrong brand.

Two complementary detectors:

  **MSER** (Maximally Stable Extremal Regions) sweeps a threshold across the
  whole intensity range and keeps regions whose area is stable across a wide
  band of it. Characters are stable in precisely that sense — a letter is a
  connected blob that persists over many thresholds — so MSER finds them
  without any prior on where text is or how it is lit.

  **Stroke Width Transform** exploits the fact that text has near-constant
  stroke width along a glyph, whereas noise, logos and packaging texture do not.
  The ratio of standard deviation to mean stroke width separates them cheaply.
"""

from __future__ import annotations

import cv2
import numpy as np

from .acquire import to_gray


def mser_regions(
    img: np.ndarray,
    *,
    min_area: int = 40,
    max_area_fraction: float = 0.05,
    delta: int = 5,
) -> list[tuple[int, int, int, int]]:
    """Detect character-like regions. Returns (x, y, w, h) boxes.

    `max_area_fraction` caps region size relative to the frame: without it MSER
    happily returns the entire strip as one very stable region, which is true
    and useless.
    """
    gray = to_gray(img)
    h, w = gray.shape
    max_area = int(h * w * max_area_fraction)

    try:
        mser = cv2.MSER_create(delta=delta, min_area=min_area, max_area=max_area)
    except TypeError:  # older OpenCV bindings use positional setters
        mser = cv2.MSER_create()
        mser.setDelta(delta)
        mser.setMinArea(min_area)
        mser.setMaxArea(max_area)

    regions, _ = mser.detectRegions(gray)

    boxes: list[tuple[int, int, int, int]] = []
    for region in regions:
        x, y, rw, rh = cv2.boundingRect(region.reshape(-1, 1, 2))
        if _plausible_character(rw, rh):
            boxes.append((x, y, rw, rh))

    return _merge_overlapping(boxes)


def _plausible_character(w: int, h: int) -> bool:
    """Geometric filter on a candidate glyph box.

    Latin characters at any font size occupy a fairly narrow aspect band. A box
    ten times wider than it is tall is a rule line or a barcode; one ten times
    taller is a packaging border.
    """
    if h == 0 or w == 0:
        return False
    aspect = w / float(h)
    return 0.08 <= aspect <= 8.0 and h >= 8


def _merge_overlapping(
    boxes: list[tuple[int, int, int, int]], *, iou_threshold: float = 0.4
) -> list[tuple[int, int, int, int]]:
    """Collapse duplicate detections.

    MSER returns nested regions for the same glyph (a character detected at
    several thresholds), so raw output is heavily redundant.
    """
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept: list[tuple[int, int, int, int]] = []

    for box in boxes:
        if not any(_iou(box, k) > iou_threshold for k in kept):
            kept.append(box)

    return kept


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0

    intersection = (ix1 - ix0) * (iy1 - iy0)
    union = aw * ah + bw * bh - intersection
    return intersection / float(union) if union else 0.0


def stroke_width_stats(mask: np.ndarray) -> tuple[float, float]:
    """Estimate (mean, std) stroke width from a binary text mask.

    Approximates the Stroke Width Transform via the distance transform: the
    distance to the nearest background pixel, evaluated along a stroke's medial
    axis, is half the local stroke width. Taking the upper quartile of the
    distance map approximates sampling the medial axis without the cost of an
    explicit skeletonisation.
    """
    if mask.max() == 0:
        return 0.0, 0.0

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    interior = dist[dist > 0]
    if interior.size < 20:
        return 0.0, 0.0

    ridge = interior[interior >= np.percentile(interior, 75)]
    widths = ridge * 2.0
    return float(widths.mean()), float(widths.std())


def is_texty(mask: np.ndarray, *, max_cv: float = 0.65) -> bool:
    """Heuristic: does this mask contain text?

    Tests the coefficient of variation of stroke width. Text is written with a
    pen or a font of near-constant width, so its CV is low; packaging texture,
    logos and noise have wildly varying "stroke" widths and a high CV.
    """
    mean, std = stroke_width_stats(mask)
    if mean <= 0:
        return False
    return (std / mean) <= max_cv


def text_mask_from_regions(
    shape: tuple[int, int], boxes: list[tuple[int, int, int, int]], *, pad: int = 2
) -> np.ndarray:
    """Rasterise detected boxes into a mask usable for cropping."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    h, w = shape[:2]
    for x, y, bw, bh in boxes:
        cv2.rectangle(
            mask,
            (max(0, x - pad), max(0, y - pad)),
            (min(w, x + bw + pad), min(h, y + bh + pad)),
            255,
            -1,
        )
    return mask


def group_into_lines(
    boxes: list[tuple[int, int, int, int]], *, y_tolerance: float = 0.6
) -> list[list[tuple[int, int, int, int]]]:
    """Group character boxes into text lines by vertical overlap.

    Lets the pipeline report *where* text was found and in what arrangement,
    which is what makes the vertical composition text along a strip edge
    detectable as its own line rather than as noise interleaved with the
    horizontal brand name.
    """
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: b[1])
    lines: list[list[tuple[int, int, int, int]]] = []

    for box in boxes:
        _, y, _, h = box
        placed = False
        for line in lines:
            ly = np.mean([b[1] for b in line])
            lh = np.mean([b[3] for b in line])
            if abs(y - ly) <= y_tolerance * max(lh, h):
                line.append(box)
                placed = True
                break
        if not placed:
            lines.append([box])

    return [sorted(line, key=lambda b: b[0]) for line in lines]
