"""
Stage 3 — specular glare removal and illumination normalisation.

This is the highest-value stage for Indian blister packs, and the clearest
example of why a DIP front-end is not redundant with a vision model. Aluminium
foil under a phone flash produces saturated specular highlights: the pixels are
clipped at 255 and the text under them is *gone from the file*. A vision model
handed that image cannot recover the characters, because it never received them.

Inpainting cannot recreate the true characters either — but it can remove the
false high-contrast edges the highlight introduces, which otherwise dominate
binarisation and boundary detection and corrupt the *rest* of the image. The
win is containment: the damage stays local instead of spreading.

Detection uses the conjunction (high Value AND low Saturation). Specular
reflection is the light source's own colour, so it is bright and desaturated;
white packaging is bright but retains some saturation. Thresholding on
brightness alone would erase every white label on the strip.
"""

from __future__ import annotations

import cv2
import numpy as np

from .acquire import to_gray, to_hsv


def specular_mask(
    img: np.ndarray,
    *,
    v_threshold: int = 245,
    s_threshold: int = 40,
    dilate_px: int = 3,
) -> np.ndarray:
    """Binary mask (uint8 0/255) of specular highlight pixels.

    The mask is dilated slightly because a blown highlight is surrounded by a
    partially-saturated halo. Inpainting only the fully-clipped core leaves a
    bright ring behind, which binarisation then reads as a stroke.
    """
    hsv = to_hsv(img)
    s, v = hsv[:, :, 1], hsv[:, :, 2]

    mask = ((v >= v_threshold) & (s <= s_threshold)).astype(np.uint8) * 255

    if dilate_px > 0 and mask.any():
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1,) * 2)
        mask = cv2.dilate(mask, k, iterations=1)

    return mask


def glare_fraction(img: np.ndarray, *, v_threshold: int = 245, s_threshold: int = 40) -> float:
    """Proportion of the frame lost to specular highlights, in [0, 1].

    Surfaced to the user: above ~0.25 the honest answer is "retake the photo",
    not a guess. This is the quality gate feeding the abstention logic.
    """
    mask = specular_mask(img, v_threshold=v_threshold, s_threshold=s_threshold, dilate_px=0)
    return float((mask > 0).sum() / mask.size)


def inpaint_glare(
    img: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    method: str = "telea",
    radius: int = 5,
    v_threshold: int = 245,
    s_threshold: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """Inpaint specular regions. Returns (result, mask_used).

    Telea marches inward from the boundary weighting known neighbours by
    distance — fast, and good for the small scattered blobs foil produces.
    Navier-Stokes propagates isophotes (lines of equal intensity) into the hole,
    which looks better on large contiguous regions but costs more.
    """
    if mask is None:
        mask = specular_mask(img, v_threshold=v_threshold, s_threshold=s_threshold)

    if not mask.any():
        return img, mask

    flag = cv2.INPAINT_NS if method == "ns" else cv2.INPAINT_TELEA
    return cv2.inpaint(img, mask, radius, flag), mask


# --- illumination normalisation ------------------------------------------


def divide_illumination(img: np.ndarray, blur_ksize: int = 51) -> np.ndarray:
    """Flatten lighting by dividing the image by a heavily blurred copy.

    Rests on the reflectance model I = R * L: a strong low-pass estimates the
    illumination L, and dividing it out leaves reflectance R. Cheap, robust, and
    the right default for the smooth lighting gradient of a handheld shot.
    """
    gray = to_gray(img)
    blur_ksize = blur_ksize | 1
    background = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    background = np.where(background == 0, 1, background).astype(np.float32)

    normalised = (gray.astype(np.float32) / background) * 128.0
    return np.clip(normalised, 0, 255).astype(np.uint8)


def multi_scale_retinex(
    img: np.ndarray, scales: tuple[int, ...] = (15, 80, 250)
) -> np.ndarray:
    """Multi-Scale Retinex.

    Single-Scale Retinex is log(I) - log(I * G_sigma): subtracting the blurred
    log-image removes the slowly-varying illumination component. One sigma
    forces a choice between dynamic-range compression (small) and colour
    constancy (large), so MSR averages several scales and gets both.
    """
    gray = to_gray(img).astype(np.float32) + 1.0
    log_img = np.log(gray)

    accum = np.zeros_like(log_img)
    for sigma in scales:
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
        accum += log_img - np.log(blurred + 1.0)
    accum /= len(scales)

    lo, hi = np.percentile(accum, 1), np.percentile(accum, 99)
    if hi - lo < 1e-6:
        return to_gray(img)

    stretched = (accum - lo) / (hi - lo) * 255.0
    return np.clip(stretched, 0, 255).astype(np.uint8)


def apply(
    img: np.ndarray,
    *,
    remove_glare: bool = True,
    glare_method: str = "telea",
    v_threshold: int = 245,
    s_threshold: int = 40,
    normalize: bool = True,
    illum_method: str = "divide",
) -> tuple[np.ndarray, dict]:
    """Run glare removal then illumination normalisation.

    Returns the processed image plus a metrics dict that flows into
    `image_quality` in the API response.
    """
    metrics: dict = {"glare_fraction": glare_fraction(img, v_threshold=v_threshold,
                                                      s_threshold=s_threshold)}
    out = img
    mask = None

    if remove_glare:
        out, mask = inpaint_glare(
            out, method=glare_method, v_threshold=v_threshold, s_threshold=s_threshold
        )
        metrics["glare_inpainted"] = bool(mask is not None and mask.any())

    if normalize:
        if illum_method == "divide":
            out = divide_illumination(out)
        elif illum_method == "retinex":
            out = multi_scale_retinex(out)
        elif illum_method == "homomorphic":
            from .frequency import homomorphic_filter

            out = homomorphic_filter(out)
        metrics["illumination_method"] = illum_method

    return out, metrics
