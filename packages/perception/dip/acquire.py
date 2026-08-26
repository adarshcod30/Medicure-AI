"""
Stage 1 — acquisition.

Decode bytes to a BGR array, honour EXIF orientation, and bound the resolution.

The EXIF step is not cosmetic: iOS and most Android cameras store the sensor
readout unrotated and record the intended rotation in a tag. PIL applies it;
`cv2.imdecode` does not. Skipping it means a large share of phone uploads arrive
90 degrees off, which quietly destroys both OCR and boundary detection.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image, ImageOps


class DecodeError(ValueError):
    """The uploaded bytes are not a decodable image."""


def decode(image_bytes: bytes, *, auto_orient: bool = True) -> np.ndarray:
    """Decode image bytes to a BGR uint8 array.

    PIL is used rather than `cv2.imdecode` specifically so that
    `ImageOps.exif_transpose` can normalise orientation before we hand the array
    to OpenCV.
    """
    if not image_bytes:
        raise DecodeError("empty image payload")

    try:
        pil = Image.open(io.BytesIO(image_bytes))
        pil.load()
    except Exception as exc:  # noqa: BLE001 - any decoder failure is the same to us
        raise DecodeError(f"could not decode image: {exc}") from exc

    if auto_orient:
        try:
            pil = ImageOps.exif_transpose(pil)
        except Exception:  # noqa: BLE001 - malformed EXIF must not fail the upload
            pass

    if pil.mode != "RGB":
        pil = pil.convert("RGB")

    rgb = np.asarray(pil, dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def limit_resolution(img: np.ndarray, max_dimension: int) -> tuple[np.ndarray, float]:
    """Downscale so the longest edge is at most `max_dimension`.

    Returns the image and the scale factor applied (1.0 if untouched), so that
    coordinates found downstream can be mapped back to the original if needed.

    INTER_AREA is the correct interpolation for downscaling — it averages over
    the source footprint rather than point-sampling, which avoids the aliasing
    that would otherwise turn fine printed text into moire.
    """
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_dimension:
        return img, 1.0

    scale = max_dimension / float(longest)
    resized = cv2.resize(
        img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA
    )
    return resized, scale


def upscale_if_tiny(img: np.ndarray, min_dimension: int = 640) -> np.ndarray:
    """Upscale very small crops before OCR.

    Tesseract's character models expect a capital-letter height of roughly
    20-30 px. A thumbnail-sized strip photo sits well under that, and cubic
    upsampling measurably recovers accuracy even though it adds no information.
    """
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest >= min_dimension:
        return img
    scale = min_dimension / float(longest)
    return cv2.resize(
        img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_CUBIC
    )


# --- colour space helpers ------------------------------------------------
# Kept here so every other stage converts consistently and no module has to
# remember whether OpenCV wants BGR or RGB.


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def to_hsv(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def to_lab(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)


def from_lab(lab: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def to_ycrcb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)


def ensure_bgr(img: np.ndarray) -> np.ndarray:
    """Promote a single-channel image to 3-channel BGR."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img
