"""
Frequency-domain processing.

Two jobs the spatial domain handles badly:

  * **Homomorphic filtering** — illumination varies slowly (low frequency),
    reflectance varies fast (high frequency), but they combine *multiplicatively*
    so no linear spatial filter separates them. Taking the logarithm turns the
    product into a sum, at which point a high-pass filter suppresses the
    illumination and keeps the reflectance. This is the textbook correction for
    a photo lit unevenly from one side, which describes most phone shots of a
    strip held in one hand.

  * **Notch filtering** — periodic interference (moire from photographing a
    screen, scanner banding) is a small number of isolated spikes in the
    spectrum. It is trivially removable there and essentially unremovable in the
    spatial domain, where it is spread across every pixel.
"""

from __future__ import annotations

import cv2
import numpy as np

from .acquire import to_gray


def _distance_grid(shape: tuple[int, int]) -> np.ndarray:
    """Euclidean distance of each frequency bin from the (centred) DC term."""
    rows, cols = shape
    cy, cx = rows / 2.0, cols / 2.0
    y = np.arange(rows).reshape(-1, 1) - cy
    x = np.arange(cols).reshape(1, -1) - cx
    return np.sqrt(y**2 + x**2)


def spectrum(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (log-magnitude spectrum for display, complex centred FFT)."""
    gray = to_gray(img).astype(np.float32)
    fft = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = 20.0 * np.log(np.abs(fft) + 1.0)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return magnitude.astype(np.uint8), fft


def butterworth(
    shape: tuple[int, int], cutoff: float, order: int = 2, *, highpass: bool = True
) -> np.ndarray:
    """Butterworth transfer function.

    Chosen over an ideal (brick-wall) filter because a sharp cutoff in the
    frequency domain is a sinc in the spatial domain, which produces visible
    ringing around every edge — and ringing next to text reads as extra strokes.
    """
    d = _distance_grid(shape)
    cutoff = max(cutoff, 1e-6)
    lowpass = 1.0 / (1.0 + (d / cutoff) ** (2 * order))
    return 1.0 - lowpass if highpass else lowpass


def homomorphic_filter(
    img: np.ndarray,
    *,
    gamma_low: float = 0.5,
    gamma_high: float = 2.0,
    cutoff: float = 32.0,
    order: int = 2,
) -> np.ndarray:
    """Suppress multiplicative illumination, amplify reflectance.

    `gamma_low` < 1 attenuates the low frequencies carrying illumination;
    `gamma_high` > 1 boosts the high frequencies carrying edges and text.
    """
    gray = to_gray(img).astype(np.float32)
    log_img = np.log1p(gray)

    fft = np.fft.fftshift(np.fft.fft2(log_img))

    hp = butterworth(gray.shape, cutoff, order, highpass=True)
    transfer = (gamma_high - gamma_low) * hp + gamma_low

    filtered = np.fft.ifft2(np.fft.ifftshift(fft * transfer))
    result = np.expm1(np.real(filtered))

    return cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def detect_periodic_peaks(
    img: np.ndarray, *, min_radius: int = 10, top_k: int = 8
) -> list[tuple[int, int]]:
    """Locate candidate periodic-noise spikes in the spectrum.

    Peaks within `min_radius` of DC are ignored — those carry the image's own
    low-frequency content, and notching them would gut it.
    """
    magnitude, _ = spectrum(img)
    rows, cols = magnitude.shape
    d = _distance_grid((rows, cols))

    searchable = magnitude.astype(np.float32).copy()
    searchable[d < min_radius] = 0

    peaks: list[tuple[int, int]] = []
    work = searchable.copy()
    for _ in range(top_k):
        _, maxval, _, maxloc = cv2.minMaxLoc(work)
        if maxval <= 0:
            break
        x, y = maxloc
        peaks.append((y, x))
        cv2.circle(work, (x, y), min_radius, 0, -1)

    return peaks


def notch_filter(
    img: np.ndarray, peaks: list[tuple[int, int]] | None = None, *, radius: int = 8
) -> np.ndarray:
    """Zero out periodic-noise spikes and their conjugate mirrors.

    A real-valued image has a Hermitian-symmetric spectrum, so every spike has a
    partner reflected through DC. Suppressing only one of the pair leaves the
    image complex and reintroduces the artefact on the inverse transform.
    """
    gray = to_gray(img).astype(np.float32)
    fft = np.fft.fftshift(np.fft.fft2(gray))

    if peaks is None:
        peaks = detect_periodic_peaks(img)
    if not peaks:
        return to_gray(img)

    rows, cols = gray.shape
    cy, cx = rows // 2, cols // 2

    mask = np.ones((rows, cols), dtype=np.float32)
    for y, x in peaks:
        cv2.circle(mask, (x, y), radius, 0, -1)
        cv2.circle(mask, (2 * cx - x, 2 * cy - y), radius, 0, -1)  # conjugate mirror

    result = np.real(np.fft.ifft2(np.fft.ifftshift(fft * mask)))
    return cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
