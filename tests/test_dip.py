"""
Tests for the digital image processing layer.

The geometric tests are the valuable ones. A wrong point ordering in
`order_points`, or transposed width/height in `target_size`, produces an image
that still *looks* broadly plausible — mirrored, or squashed — and a visual
check will pass it. Warping by a known homography and asserting the inverse
recovers the original catches exactly that class of defect.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from packages.perception.dip import binarize, denoise, edges, glare, morphology, segment
from packages.perception.dip.config import ABLATION_LADDER, DipConfig
from packages.perception.dip.pipeline import run
from packages.perception.dip.quality import assess
from packages.perception.dip.rectify import deskew, estimate_skew, four_point_transform, rotate
from tests import synthetic


# --- helpers --------------------------------------------------------------


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation of two same-size greyscale images."""
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32) if a.ndim == 3 else a.astype(np.float32)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32) if b.ndim == 3 else b.astype(np.float32)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    return float(((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std() + 1e-9))


def quad_iou(a: np.ndarray, b: np.ndarray, shape: tuple[int, int]) -> float:
    """IoU of two quadrilaterals, computed by rasterising them."""
    ma = np.zeros(shape, np.uint8)
    mb = np.zeros(shape, np.uint8)
    cv2.fillPoly(ma, [a.astype(np.int32)], 255)
    cv2.fillPoly(mb, [b.astype(np.int32)], 255)
    inter = np.logical_and(ma, mb).sum()
    union = np.logical_or(ma, mb).sum()
    return float(inter / union) if union else 0.0


# --- geometry -------------------------------------------------------------


def test_order_points_is_orientation_invariant():
    """The same four corners must order identically regardless of input order."""
    quad = np.array([[10, 20], [200, 30], [190, 150], [5, 140]], dtype=np.float32)
    expected = segment.order_points(quad)

    rng = np.random.default_rng(0)
    for _ in range(8):
        shuffled = quad[rng.permutation(4)]
        np.testing.assert_allclose(segment.order_points(shuffled), expected, atol=1e-4)


def test_four_point_transform_inverts_a_known_homography():
    """Warp by a known homography, rectify with the known corners, recover the original.

    This isolates `four_point_transform` from boundary detection: the quad is
    handed in exactly, so any error is in the warp itself.
    """
    clean = synthetic.make_strip()
    h, w = clean.shape[:2]

    warped, matrix = synthetic.apply_perspective(clean, strength=0.18)

    # Where the original image's own corners landed under the warp.
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    landed = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), matrix).reshape(4, 2)

    recovered = four_point_transform(warped, landed)

    assert ncc(clean, recovered) > 0.90, "rectified image does not match the original"


def layout_ncc(a: np.ndarray, b: np.ndarray, blur: int = 15) -> float:
    """NCC of heavily blurred images — compares layout, not pixel registration.

    Raw NCC is unusable for this comparison. Text is high-frequency, so a
    three-pixel misregistration on a twenty-pixel glyph decorrelates it almost
    completely, and boundary detection is only ever accurate to a few pixels.
    Blurring first compares where the blocks of text *are*, which is the
    property rectification is supposed to restore.
    """
    a_blur = cv2.GaussianBlur(a, (blur | 1, blur | 1), 0)
    b_blur = cv2.GaussianBlur(b, (blur | 1, blur | 1), 0)
    return ncc(a_blur, b_blur)


def test_rectification_measurably_improves_on_the_distorted_image():
    """End-to-end geometry: detect the quad, rectify, and confirm it helped.

    Asserted as an *improvement over the distorted input* rather than against an
    absolute threshold. That is both the more robust test and the more honest
    claim: the ablation table's "+rectify" row is exactly this delta, so the
    test checks the thing the project actually asserts.
    """
    clean = synthetic.make_strip()
    warped, matrix = synthetic.apply_perspective(clean, strength=0.15)

    quad, method = segment.find_packet_quad(warped)
    assert quad is not None, f"no packet boundary found (method={method})"

    rectified = four_point_transform(warped, quad)

    h, w = clean.shape[:2]
    true_packet = clean[50 : h - 50, 60 : w - 60]

    before = layout_ncc(true_packet, warped)
    after = layout_ncc(true_packet, rectified)

    assert after > before, f"rectification made it worse: {before:.3f} -> {after:.3f}"
    assert after > 0.60, f"rectified layout still poorly aligned: {after:.3f}"


def test_boundary_detection_localises_the_packet():
    """The detected quad should substantially overlap the true packet."""
    clean = synthetic.make_strip()
    h, w = clean.shape[:2]
    warped, matrix = synthetic.apply_perspective(clean, strength=0.15)

    packet = np.array(
        [[60, 50], [w - 60, 50], [w - 60, h - 50], [60, h - 50]], dtype=np.float32
    )
    true_quad = cv2.perspectiveTransform(packet.reshape(-1, 1, 2), matrix).reshape(4, 2)

    found, method = segment.find_packet_quad(warped)
    assert found is not None, f"no boundary found (method={method})"

    assert quad_iou(found, true_quad, (h, w)) > 0.80


@pytest.mark.parametrize("angle", [-11.0, -8.0, -3.5, 4.0, 11.0])
def test_skew_estimation_recovers_a_known_rotation(angle: float):
    """Rotate by a known angle; the estimator should measure it back, signed."""
    rotated = rotate(synthetic.make_strip(), angle)
    estimated, method = estimate_skew(rotated)

    assert method != "none", "skew estimator produced no estimate"
    assert abs(estimated - angle) < 1.0, f"got {estimated}, expected {angle}"


@pytest.mark.parametrize("angle", [-9.0, 6.5])
def test_deskew_round_trip_actually_straightens(angle: float):
    """Deskew must reduce the skew, not double it.

    A sign error in the correction is completely silent — the output is still a
    rotated image that looks superficially fine — so the only way to catch it is
    to re-measure the result. Doubling would give ~2x the input angle here.
    """
    rotated = rotate(synthetic.make_strip(), angle)

    corrected, metrics = deskew(rotated)
    assert metrics["skew_corrected"] is True

    residual, _ = estimate_skew(corrected)
    assert abs(residual) < 1.0, f"residual skew {residual} after correcting {angle}"


def test_skew_estimate_is_refused_beyond_the_limit():
    """A large angle must not be silently 'corrected'.

    Beyond the limit the estimator has almost certainly locked onto packaging
    graphics rather than a text baseline, and rotating by a wrong large angle is
    far more damaging than leaving the image alone.
    """
    rotated = rotate(synthetic.make_strip(), 45.0)
    estimated, _ = estimate_skew(rotated, max_angle=20.0)
    assert abs(estimated) <= 20.0


# --- glare ----------------------------------------------------------------


def test_glare_is_detected_and_reduced():
    clean = synthetic.make_strip()
    glared = synthetic.add_glare(clean, n_spots=4, max_radius=110, seed=1)

    before = glare.glare_fraction(glared)
    assert before > 0.02, "synthetic glare was not detectable"

    repaired, mask = glare.inpaint_glare(glared)
    after = glare.glare_fraction(repaired)

    assert after < before * 0.5, f"inpainting barely helped: {before:.3f} -> {after:.3f}"


def test_specular_mask_spares_bright_but_saturated_colour():
    """Bright saturated colour is packaging, not a specular highlight.

    Guards the conjunction in `specular_mask`. Thresholding on brightness alone
    would erase every white-on-colour label on the strip.
    """
    img = np.zeros((100, 100, 3), np.uint8)
    img[:, :] = (0, 0, 255)  # saturated red, maximum value in one channel
    assert glare.specular_mask(img).sum() == 0


# --- denoising ------------------------------------------------------------


def test_noise_estimator_tracks_injected_noise():
    clean = synthetic.make_strip()
    sigmas = [0.0, 8.0, 20.0]
    estimates = [denoise.estimate_noise_sigma(synthetic.add_noise(clean, s, seed=2)) for s in sigmas]

    assert estimates[0] < estimates[1] < estimates[2], f"not monotonic: {estimates}"


def test_auto_denoise_picks_median_for_impulse_noise():
    speckled = synthetic.add_impulse(synthetic.make_strip(), fraction=0.02, seed=3)
    assert denoise.select_method(speckled) == "median"


# --- edges ----------------------------------------------------------------


def test_gradient_canny_beats_median_canny_on_a_bright_image():
    """Regression test for a real bug.

    The median-sigma rule computes `upper = (1 + sigma) * median`. On a bright
    image (a strip fills the frame; median ~238) that is ~316, clamped to 255 —
    above every gradient present. The strong-edge set comes out empty and Canny
    returns almost nothing. The gradient-percentile rule is immune because it
    thresholds on the gradient distribution rather than on brightness.
    """
    bright = synthetic.degraded_strip()[0]

    median_density = edges.edge_density(edges.auto_canny(bright, method="median"))
    gradient_density = edges.edge_density(edges.auto_canny(bright, method="gradient"))

    assert gradient_density > median_density * 3, (
        f"gradient rule did not clearly win: {gradient_density:.4f} vs {median_density:.4f}"
    )


# --- binarisation ---------------------------------------------------------


def test_sauvola_survives_an_illumination_gradient_that_defeats_otsu():
    """The textbook justification for local thresholding, asserted.

    Under a strong lighting ramp the global histogram stops being bimodal, so
    Otsu must sacrifice one end of the image. Sauvola thresholds locally and
    keeps text across the whole frame.
    """
    lit = synthetic.add_illumination_gradient(synthetic.make_strip(), strength=0.72)

    otsu_mask = binarize.otsu(lit)
    sauvola_mask = binarize.sauvola(lit)

    # Compare ink recovered in the darkened third against the bright third.
    w = lit.shape[1]
    def balance(mask):
        bright = (mask[:, : w // 3] > 0).mean()
        dark = (mask[:, -w // 3 :] > 0).mean()
        return min(bright, dark) / (max(bright, dark) + 1e-9)

    assert balance(sauvola_mask) > balance(otsu_mask)


def test_ink_coverage_is_in_a_plausible_band_for_text():
    mask = binarize.sauvola(synthetic.make_strip())
    assert 0.005 < binarize.ink_coverage(mask) < 0.40


# --- morphology -----------------------------------------------------------


def test_tophat_isolates_text_from_an_illumination_gradient():
    lit = synthetic.add_illumination_gradient(synthetic.make_strip(), strength=0.7)
    result = morphology.black_hat(lit, size=15)

    # The background should collapse to near zero; strokes should remain bright.
    assert result.mean() < 40
    assert result.max() > 90


def test_text_polarity_detects_dark_on_light():
    assert morphology.text_polarity(synthetic.make_strip()) == "dark_on_light"


# --- quality gate ---------------------------------------------------------


def test_quality_gate_flags_a_severely_blurred_image():
    blurred = synthetic.add_blur(synthetic.make_strip(), ksize=41)
    report = assess(blurred)

    assert report.verdict in {"poor", "unusable"}
    assert report.advice, "a degraded image must come with actionable advice"


def test_quality_gate_passes_a_clean_image():
    report = assess(synthetic.make_strip())
    assert report.verdict == "good"
    assert not report.should_abstain


def test_quality_gate_abstains_on_a_hopeless_image():
    hopeless = synthetic.add_blur(synthetic.make_strip(), ksize=61)
    hopeless = synthetic.add_glare(hopeless, n_spots=9, max_radius=220, seed=4)

    report = assess(hopeless)
    assert report.should_abstain
    assert report.reasons


# --- pipeline -------------------------------------------------------------


def test_pipeline_runs_every_ablation_rung():
    """Every rung of the ablation ladder must execute without error.

    `eval/bench_ocr.py` iterates this list, so a config that crashes would take
    the whole benchmark down.
    """
    image = synthetic.encode(synthetic.degraded_strip()[0])

    for name, config in ABLATION_LADDER:
        result = run(image, config)
        assert result.renditions, f"rung '{name}' produced no renditions"
        assert result.quality.verdict in {"good", "degraded", "poor", "unusable"}


def test_pipeline_respects_the_rendition_cap():
    config = DipConfig.full().with_(max_renditions=4)
    result = run(synthetic.encode(synthetic.degraded_strip()[0]), config)
    # The cap bounds binarised renditions; the raw greyscale is always appended.
    assert len(result.renditions) <= config.max_renditions + 1


def test_raw_preset_does_no_processing():
    result = run(synthetic.encode(synthetic.make_strip()), DipConfig.raw())
    assert len(result.renditions) == 1
    assert result.renditions[0].name == "gray:none:rot0"
    assert result.metrics["denoise_method"] == "none"


def test_quality_is_measured_before_restoration():
    """Regression test for a real bug.

    Quality was originally assessed after glare inpainting, so a photo that was
    12% blown-out reported 0% glare — the pipeline was grading its own output
    and would have waved through images it should have refused.
    """
    glared = synthetic.add_glare(synthetic.make_strip(), n_spots=5, max_radius=120, seed=5)
    result = run(synthetic.encode(glared), DipConfig.full())

    assert result.quality.glare_fraction > 0.02, (
        "glare fraction was measured after inpainting and reads as clean"
    )


def test_dump_stages_captures_intermediates():
    config = DipConfig.full().with_(dump_stages=True)
    result = run(synthetic.encode(synthetic.degraded_strip()[0]), config)

    assert len(result.stages) > 5
    assert any("rendition" in k for k in result.stages)
