"""
Stage 11 — image quality assessment and routing.

This module is where the DIP layer serves the project's central thesis. The
system's differentiator is that it declines to answer when it should not answer,
and an image-quality gate is the earliest and cheapest place to make that
judgement: if the photo does not contain the information, no downstream
component can supply it, and every component after this point would be guessing.

It also makes the refusal *useful*. "I could not read this" is a dead end;
"73% of this photo is blown-out glare — lay the strip flat and turn the flash
off" is an instruction the user can act on and immediately retry. The metrics
here are what turn one into the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from .acquire import to_gray
from .glare import glare_fraction

Verdict = Literal["good", "degraded", "poor", "unusable"]

# Thresholds calibrated against typical 8-12 MP phone captures downscaled to
# 2000 px. Re-fit these against your own labelled photo set once it exists —
# eval/bench_ocr.py reports the CER achieved in each band, which is the
# evidence you would tune them on.
BLUR_UNUSABLE = 30.0
BLUR_POOR = 80.0
BLUR_DEGRADED = 150.0

GLARE_UNUSABLE = 0.45
GLARE_POOR = 0.25
GLARE_DEGRADED = 0.10

TEXT_CONTRAST_POOR = 7.0
TEXT_CONTRAST_DEGRADED = 12.0

MIN_USABLE_DIMENSION = 400


@dataclass
class QualityReport:
    """Measured image quality plus a routing decision."""

    blur_variance: float
    glare_fraction: float
    rms_contrast: float
    text_contrast: float
    width: int
    height: int
    skew_deg: float = 0.0

    verdict: Verdict = "good"
    reasons: list[str] = field(default_factory=list)
    advice: list[str] = field(default_factory=list)

    use_vision_fallback: bool = False
    should_abstain: bool = False

    def to_dict(self) -> dict:
        return {
            "blur_variance": round(self.blur_variance, 2),
            "glare_fraction": round(self.glare_fraction, 4),
            "rms_contrast": round(self.rms_contrast, 2),
            "text_contrast": round(self.text_contrast, 2),
            "resolution": [self.width, self.height],
            "skew_deg": round(self.skew_deg, 2),
            "verdict": self.verdict,
            "reasons": self.reasons,
            "advice": self.advice,
            "use_vision_fallback": self.use_vision_fallback,
            "should_abstain": self.should_abstain,
        }


def blur_variance(img: np.ndarray) -> float:
    """Variance of the Laplacian — the standard no-reference focus measure.

    The Laplacian responds to intensity discontinuities. A sharp image has many
    strong edges and so a high variance of response; a blurred one has had those
    discontinuities smoothed away and its variance collapses.

    It is scale-dependent, which is exactly why `acquire.limit_resolution` runs
    before this: comparing raw values across images of different sizes would be
    meaningless, and fixing the working resolution first makes the thresholds
    above transferable.
    """
    gray = to_gray(img)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def rms_contrast(img: np.ndarray) -> float:
    """Root-mean-square contrast — the standard deviation of intensity.

    Reported for reference, but *not* used to gate. See `text_contrast` for why.
    """
    return float(to_gray(img).std())


def text_contrast(img: np.ndarray, *, sigma: float = 12.0) -> float:
    """Contrast at text scale: the standard deviation of the high-pass image.

    This replaces RMS contrast as the gating metric, and the reason is worth
    recording. RMS contrast is the standard deviation over the *whole* image,
    which on a medicine strip is dominated by the large flat regions of
    packaging and by any lighting gradient across them. Measured on our own test
    images it ranks them backwards: a badly degraded photo scored 43.2 — higher
    than a clean one at 27.8 — because glare and a lighting ramp inflate the
    global standard deviation. It was answering "how much does brightness vary
    across this photo", when the question is "how legible is the text".

    Subtracting a Gaussian-blurred copy removes everything varying more slowly
    than `sigma`, leaving modulation at stroke scale. On the same images that
    ranks them correctly: clean 21.2, degraded 11.1, faded print 6.8, blurred
    beyond use 5.2.

    Known weakness: high-frequency sensor noise also lives at this scale and
    inflates the measure. That is why the pipeline denoises before assessing.
    """
    gray = to_gray(img).astype(np.float32)
    high_pass = gray - cv2.GaussianBlur(gray, (0, 0), sigma)
    return float(high_pass.std())


def assess(img: np.ndarray, *, skew_deg: float = 0.0) -> QualityReport:
    """Measure quality and decide how to route the image.

    Routing has three outcomes:
      * OCR normally
      * OCR, and also call the vision model to transcribe (results are fused)
      * abstain and ask for a retake

    Note what the vision fallback is *for*: transcription only. A degraded image
    is precisely where an unconstrained vision model is most likely to guess a
    plausible brand name from partial visual cues, which is the failure mode
    this whole architecture exists to prevent.
    """
    h, w = img.shape[:2]
    report = QualityReport(
        blur_variance=blur_variance(img),
        glare_fraction=glare_fraction(img),
        rms_contrast=rms_contrast(img),
        text_contrast=text_contrast(img),
        width=w,
        height=h,
        skew_deg=skew_deg,
    )

    severity = 0  # 0 good, 1 degraded, 2 poor, 3 unusable

    # --- focus ---
    if report.blur_variance < BLUR_UNUSABLE:
        severity = max(severity, 3)
        report.reasons.append(f"severely out of focus (blur score {report.blur_variance:.0f})")
        report.advice.append("Hold the phone steady and tap the screen to focus before shooting.")
    elif report.blur_variance < BLUR_POOR:
        severity = max(severity, 2)
        report.reasons.append(f"out of focus (blur score {report.blur_variance:.0f})")
        report.advice.append("Move slightly further away and let the camera refocus.")
    elif report.blur_variance < BLUR_DEGRADED:
        severity = max(severity, 1)
        report.reasons.append("slightly soft focus")

    # --- glare ---
    if report.glare_fraction > GLARE_UNUSABLE:
        severity = max(severity, 3)
        report.reasons.append(f"{report.glare_fraction:.0%} of the image is blown-out glare")
        report.advice.append("Turn the flash off and shoot in indirect daylight.")
    elif report.glare_fraction > GLARE_POOR:
        severity = max(severity, 2)
        report.reasons.append(f"heavy glare on {report.glare_fraction:.0%} of the image")
        report.advice.append("Tilt the strip slightly so the light does not reflect straight back.")
    elif report.glare_fraction > GLARE_DEGRADED:
        severity = max(severity, 1)
        report.reasons.append("some reflective glare")

    # --- contrast at text scale ---
    if report.text_contrast < TEXT_CONTRAST_POOR:
        severity = max(severity, 2)
        report.reasons.append("very low contrast between text and packaging")
        report.advice.append("Increase the lighting, or photograph the printed side of the pack.")
    elif report.text_contrast < TEXT_CONTRAST_DEGRADED:
        severity = max(severity, 1)
        report.reasons.append("low text contrast")

    # --- resolution ---
    if min(h, w) < MIN_USABLE_DIMENSION:
        severity = max(severity, 2)
        report.reasons.append(f"image is small ({w}x{h})")
        report.advice.append("Fill more of the frame with the strip, or send the original photo.")

    report.verdict = ("good", "degraded", "poor", "unusable")[severity]

    # Route. Vision transcription is worth its cost from "degraded" upward,
    # where Tesseract starts dropping characters but the image still carries
    # recoverable text.
    report.use_vision_fallback = severity >= 1
    report.should_abstain = severity >= 3

    if report.should_abstain and not report.advice:
        report.advice.append("Please retake the photo in better light with the strip laid flat.")

    return report


def summarise_for_user(report: QualityReport) -> str:
    """One-sentence, actionable explanation of a quality failure."""
    if report.verdict == "good":
        return "Image quality is good."
    problems = "; ".join(report.reasons) or "the image is hard to read"
    fix = " ".join(report.advice)
    return f"This photo is hard to read — {problems}. {fix}".strip()
