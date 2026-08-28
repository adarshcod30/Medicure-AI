"""
Orientation detection by cheap OCR probing.

A photographed medicine strip arrives at an arbitrary multiple of 90 degrees.
It is not a document: there is no page, no margin and no reading order to
appeal to, and people photograph a strip whichever way it is lying. In a set of
twelve real phone photos, four were rotated 90 degrees and one was upside down
— a third of the set.

**Tesseract's own OSD mode does not work here.** It is the obvious tool and it
was tried first: `image_to_osd` failed outright on ten of those twelve with
"Too few characters" or an invalid-resolution warning, and on the two where it
did run it reported orientation confidence 0.4 and 6.7 while identifying the
script as Fraktur and Cyrillic. OSD needs a paragraph of continuous text to
vote on; a blister strip has scattered fragments.

So orientation is found by trying all four and keeping the best. What makes
that cheap rather than wasteful is doing it on a **downscaled greyscale copy**
with a single fast OCR pass. The probe costs a fraction of one full-resolution
pass, and once the angle is known the expensive pipeline runs once, at the
right orientation, instead of four times at every orientation.

180 degrees matters as much as 90. The original rendition fan-out tried
(0, 90, 270) and omitted 180 — so an upside-down strip, which is simply what
you get when a strip is lying the other way round, could never be read at all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from concurrent.futures import ThreadPoolExecutor, as_completed

from .dip import acquire
from .tesseract_engine import (
    TESSERACT_AVAILABLE,
    _available_cpus,
    _run_tesseract,
    _score_words,
)

logger = logging.getLogger(__name__)

ROTATIONS = (0, 90, 180, 270)

PROBE_MAX_DIMENSION = 1000
"""Probe resolution. Large enough that headline text is legible — which is all
the probe needs, since it is choosing between four options, not transcribing."""

PROBE_PSM = 11
"""Sparse text. The probe should find scattered fragments, not assume a block."""


@dataclass
class OrientationResult:
    angle: int
    """Degrees the image must be rotated **counter-clockwise** to be upright."""

    confidence: float
    """Winning score divided by the runner-up. 1.0 means a tie — no evidence."""

    scores: dict[int, float]
    detected: bool

    def to_dict(self) -> dict:
        return {
            "angle": self.angle,
            "confidence": round(self.confidence, 3),
            "scores": {str(k): round(v, 1) for k, v in self.scores.items()},
            "detected": self.detected,
        }


def _rotate90(image: np.ndarray, angle: int) -> np.ndarray:
    """Exact 90-degree rotation. Lossless, unlike an affine warp."""
    if angle == 0:
        return image
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)


def detect(
    image: np.ndarray, *, min_confidence: float = 1.25, lang: str = "eng"
) -> OrientationResult:
    """Find the rotation that makes `image` upright.

    `min_confidence` is a *ratio* between the best and second-best score, not an
    absolute threshold. Absolute OCR scores vary hugely with how much text a
    strip carries, so a fixed cutoff would work on a dense carton and reject a
    sparse foil. The ratio asks the only question that matters: is one
    orientation clearly better than the alternatives?

    Below the threshold the image is left alone. An unjustified 90-degree
    rotation is far more damaging than none, because every downstream stage then
    operates on sideways text.
    """
    if not TESSERACT_AVAILABLE:
        return OrientationResult(0, 1.0, {}, detected=False)

    probe, _ = acquire.limit_resolution(acquire.to_gray(image), PROBE_MAX_DIMENSION)

    # Four independent Tesseract probes, run concurrently. This was 3.79s of a
    # 4.34s DIP pass — 87% of it — because each probe waits on its own
    # subprocess. Pool size comes from the CONTAINER's allocation, not the
    # host's core count: sizing from os.cpu_count() on Cloud Run oversubscribed
    # a 4-vCPU service badly enough to make parallel OCR slower than sequential.
    #
    # Order does not matter here the way it does in read_renditions — scores go
    # into a dict keyed by angle and the winner is chosen by a sort afterwards,
    # so completion order cannot change the result.
    scores: dict[int, float] = {}
    workers = min(len(ROTATIONS), _available_cpus())
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_run_tesseract, _rotate90(probe, angle), PROBE_PSM, lang): angle
            for angle in ROTATIONS
        }
        for future in as_completed(futures):
            angle = futures[future]
            try:
                scores[angle] = _score_words(future.result())
            except Exception:  # noqa: BLE001 — a failed probe is just a bad angle
                logger.warning("orientation probe at %s degrees failed", angle)
                scores[angle] = 0.0

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_angle, best_score = ordered[0]
    runner_up = ordered[1][1] if len(ordered) > 1 else 0.0

    if best_score <= 0:
        return OrientationResult(0, 1.0, scores, detected=False)

    confidence = best_score / runner_up if runner_up > 0 else float("inf")

    if best_angle == 0 or confidence < min_confidence:
        return OrientationResult(0, confidence, scores, detected=False)

    return OrientationResult(best_angle, confidence, scores, detected=True)


def correct(
    image: np.ndarray, *, min_confidence: float = 1.25, lang: str = "eng"
) -> tuple[np.ndarray, OrientationResult]:
    """Detect orientation and apply it. Returns (upright image, result)."""
    result = detect(image, min_confidence=min_confidence, lang=lang)
    if not result.detected:
        return image, result
    return _rotate90(image, result.angle), result
