"""
Synthetic medicine-strip generator.

Produces images with *known* ground truth (text content, and the exact
homography applied), which makes two things testable that otherwise are not:

  * geometric correctness — warp by a known homography, rectify, and assert the
    result comes back within tolerance. A visual check cannot catch a subtly
    wrong point ordering; this can.
  * degradation response — apply a measured amount of glare, blur or noise and
    assert the quality gate reports it and routes accordingly.

This is not a substitute for the ~300 real photos the eval set needs. Synthetic
defects are cleaner than real ones and a pipeline can overfit to them. It is a
substitute for waiting on that shoot before any of this can be tested at all.
"""

from __future__ import annotations

import cv2
import numpy as np

DEFAULT_LINES = [
    "AUGMENTIN 625 DUO",
    "Amoxycillin 500mg +",
    "Clavulanic Acid 125mg",
    "Tablets IP",
    "Schedule H1",
    "B.No. AX2291  Exp 08/27",
    "M.R.P. Rs. 223.42",
]


def make_strip(
    lines: list[str] | None = None,
    *,
    width: int = 900,
    height: int = 500,
    bg: tuple[int, int, int] = (238, 238, 235),
    fg: tuple[int, int, int] = (25, 25, 30),
    font_scale: float = 0.85,
) -> np.ndarray:
    """Render a clean, fronto-parallel mock strip on a neutral background."""
    lines = lines or DEFAULT_LINES

    canvas = np.full((height, width, 3), 200, dtype=np.uint8)

    # The packet itself, inset so there is a real background to segment from.
    x0, y0, x1, y1 = 60, 50, width - 60, height - 50
    cv2.rectangle(canvas, (x0, y0), (x1, y1), bg, -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (150, 150, 150), 2)

    y = y0 + 55
    for i, line in enumerate(lines):
        scale = font_scale * (1.25 if i == 0 else 0.75)
        thickness = 2 if i == 0 else 1
        cv2.putText(
            canvas, line, (x0 + 30, y), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thickness, cv2.LINE_AA
        )
        y += int(48 * (1.2 if i == 0 else 0.85))

    return canvas


def apply_perspective(
    img: np.ndarray, strength: float = 0.18
) -> tuple[np.ndarray, np.ndarray]:
    """Warp by a known homography. Returns (warped, homography).

    `strength` is the corner displacement as a fraction of width — 0.18 is a
    pronounced but entirely realistic handheld angle.
    """
    h, w = img.shape[:2]
    dx = w * strength

    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    dst = np.array(
        [[dx, dx * 0.4], [w - 1 - dx * 0.3, 0], [w - 1, h - 1 - dx * 0.2], [dx * 0.5, h - 1]],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, matrix, (w, h), borderValue=(190, 190, 190))
    return warped, matrix


def add_glare(
    img: np.ndarray, *, n_spots: int = 3, max_radius: int = 90, seed: int = 0
) -> np.ndarray:
    """Add saturated specular highlights.

    Drawn as blurred white discs then clipped, which reproduces the real
    structure: a fully-clipped core with a partially-saturated halo. That halo
    is exactly what the dilation in `specular_mask` exists to catch.
    """
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]

    overlay = np.zeros((h, w), dtype=np.float32)
    for _ in range(n_spots):
        cx = int(rng.integers(w // 6, w * 5 // 6))
        cy = int(rng.integers(h // 6, h * 5 // 6))
        r = int(rng.integers(max_radius // 2, max_radius))
        cv2.circle(overlay, (cx, cy), r, 255, -1)

    overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=max_radius / 3.0)
    overlay = (overlay / max(overlay.max(), 1e-6)) * 320.0  # overshoot so cores clip

    out = img.astype(np.float32) + overlay[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def add_blur(img: np.ndarray, ksize: int = 9) -> np.ndarray:
    return cv2.GaussianBlur(img, (ksize | 1, ksize | 1), 0)


def add_noise(img: np.ndarray, sigma: float = 12.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = img.astype(np.float32) + rng.normal(0, sigma, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_impulse(img: np.ndarray, fraction: float = 0.01, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = img.copy()
    n = int(img.shape[0] * img.shape[1] * fraction)
    ys = rng.integers(0, img.shape[0], n)
    xs = rng.integers(0, img.shape[1], n)
    out[ys[: n // 2], xs[: n // 2]] = 0
    out[ys[n // 2:], xs[n // 2:]] = 255
    return out


def add_illumination_gradient(img: np.ndarray, strength: float = 0.55) -> np.ndarray:
    """Multiply by a linear ramp — the classic 'lit from one side' failure that
    defeats global Otsu and motivates Sauvola."""
    h, w = img.shape[:2]
    ramp = np.linspace(1.0, 1.0 - strength, w, dtype=np.float32)
    field = np.tile(ramp, (h, 1))
    out = img.astype(np.float32) * field[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def degraded_strip(
    *,
    perspective: float = 0.18,
    glare_spots: int = 3,
    blur: int = 5,
    noise_sigma: float = 10.0,
    gradient: float = 0.5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """A strip with every defect the pipeline targets. Returns (image, homography)."""
    img = make_strip()
    img, matrix = apply_perspective(img, perspective)
    if gradient:
        img = add_illumination_gradient(img, gradient)
    if glare_spots:
        img = add_glare(img, n_spots=glare_spots, seed=seed)
    if blur:
        img = add_blur(img, blur)
    if noise_sigma:
        img = add_noise(img, noise_sigma, seed=seed)
    return img, matrix


def encode(img: np.ndarray, ext: str = ".png") -> bytes:
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError("failed to encode synthetic image")
    return buf.tobytes()
