"""
Stage 7 — geometric rectification.

This is the stage that most clearly justifies a DIP front-end. A photo taken at
an angle projects the packet's rectangle to a trapezium: characters shrink and
shear toward the far edge, baselines converge, and stroke width varies across
the line. OCR degrades sharply under that, and no amount of contrast work fixes
it, because the problem is geometry rather than intensity.

A homography inverts the projection exactly. Four point correspondences pin down
the 8 degrees of freedom of a planar projective transform, and a medicine strip
is planar enough for that to hold. This is genuine information recovery: the
characters really do come back.

Two steps, deliberately separate:
  * `four_point_transform` — the projective warp, needs the quad from `segment`
  * `deskew` — residual in-plane rotation, works with no quad at all
"""

from __future__ import annotations

import cv2
import numpy as np

from .acquire import to_gray
from .edges import auto_canny
from .segment import order_points


def _edge_length(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def target_size(quad: np.ndarray) -> tuple[int, int]:
    """Output dimensions for a rectified quad.

    Each side is measured on both of its parallel edges and the larger taken:
    the longer edge is the one nearer the camera and therefore the one that was
    *least* compressed by the projection, so it best preserves the true
    resolution. Using the shorter edge would bake the foreshortening loss in
    permanently.
    """
    tl, tr, br, bl = quad
    width = max(_edge_length(br, bl), _edge_length(tr, tl))
    height = max(_edge_length(tr, br), _edge_length(tl, bl))
    return max(int(round(width)), 1), max(int(round(height)), 1)


def four_point_transform(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Warp `quad` onto a fronto-parallel rectangle.

    `getPerspectiveTransform` solves for the 3x3 homography from the four
    correspondences; `warpPerspective` applies it. INTER_CUBIC is used because
    parts of the image are being *enlarged* (the compressed far edge), and cubic
    resampling reconstructs strokes noticeably better than bilinear there.
    """
    quad = order_points(quad)
    width, height = target_size(quad)

    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32
    )

    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), destination)
    return cv2.warpPerspective(
        img, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def estimate_skew_hough(img: np.ndarray, *, max_angle: float = 20.0) -> float | None:
    """Estimate in-plane rotation from near-horizontal Hough lines.

    Text lines, rule lines and the packet's own edges are all approximately
    horizontal in a correctly-oriented image. Their dominant deviation from
    horizontal is the skew. Only lines within `max_angle` are counted, so the
    vertical composition text printed along many Indian strips does not drag the
    estimate 90 degrees off.

    The *median* angle is returned rather than the mean, because a handful of
    diagonal packaging graphics would otherwise shift a mean noticeably.
    """
    edges = auto_canny(img)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 360, threshold=100, minLineLength=img.shape[1] // 4,
        maxLineGap=20,
    )
    if lines is None or len(lines) == 0:
        return None

    # OpenCV 4 returns shape (N, 1, 4); OpenCV 5 returns (N, 4). Reshaping
    # rather than indexing `[:, 0]` keeps this working on both.
    lines = np.asarray(lines).reshape(-1, 4)

    # Discard lines hugging the frame. After a warp or rotation the canvas edges
    # carry border-fill artefacts — perfectly straight, perfectly axis-aligned,
    # and long enough to dominate the Hough vote. Left in, they drag every
    # estimate toward zero and the deskew silently stops working.
    h, w = img.shape[:2]
    margin = max(4, int(min(h, w) * 0.02))

    angles: list[float] = []
    for x1, y1, x2, y2 in lines:
        near_border = (
            (x1 <= margin and x2 <= margin)
            or (x1 >= w - margin and x2 >= w - margin)
            or (y1 <= margin and y2 <= margin)
            or (y1 >= h - margin and y2 >= h - margin)
        )
        if near_border:
            continue

        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
        if abs(angle) <= max_angle:
            angles.append(float(angle))

    if not angles:
        return None
    return float(np.median(angles))


def _text_mask_for_skew(img: np.ndarray, *, max_width: int = 800) -> np.ndarray | None:
    """Downscaled binary text mask used by the projection-profile estimator."""
    gray = to_gray(img)

    h, w = gray.shape[:2]
    if max(h, w) > max_width:
        scale = max_width / float(max(h, w))
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # Sauvola rather than Otsu: the mask must survive a lighting gradient, and
    # a global threshold would drop the text on the darker side entirely,
    # biasing the profile toward whichever half stayed legible.
    from .binarize import sauvola

    mask = sauvola(gray, window=25, k=0.2, invert=True)

    from .morphology import remove_small_components

    mask = remove_small_components(mask, min_area=6)
    return mask if mask.any() else None


def _profile_score(mask: np.ndarray) -> float:
    """Score how well-aligned horizontal text lines are in `mask`.

    Sums each row to form a horizontal projection profile. When text lines are
    exactly horizontal the profile is sharply peaked — rows through a line of
    text are dense, rows through the gap between lines are empty. As the image
    rotates, each text line smears across many rows and the profile flattens.

    The score is the sum of squared differences between adjacent rows, which
    rewards sharp transitions between text rows and gaps. This is more
    discriminative than the variance of the profile, because variance is also
    raised by a slow overall density gradient that has nothing to do with skew.
    """
    profile = mask.sum(axis=1, dtype=np.float64)
    return float((np.diff(profile) ** 2).sum())


def estimate_skew_projection(
    img: np.ndarray, *, max_angle: float = 20.0, coarse_step: float = 1.0
) -> float | None:
    """Estimate skew by maximising the horizontal projection profile score.

    The standard document-deskew technique, and the right primary method here.
    Unlike Hough it needs no long straight lines — which medicine strips do not
    reliably have — and unlike a single `minAreaRect` fit it is not thrown off
    by one stray blob, because every text pixel contributes a vote.

    Two passes: a coarse sweep at `coarse_step` across the whole range, then a
    fine sweep at 0.1 degrees around the winner. A single fine sweep over the
    full range would cost roughly ten times as much for the same answer.
    """
    mask = _text_mask_for_skew(img)
    if mask is None:
        return None

    def best_angle(candidates: np.ndarray) -> tuple[float, float]:
        best_a, best_s = 0.0, -1.0
        for angle in candidates:
            # borderValue=0 so the padding introduced by rotation contributes no
            # ink, and cannot manufacture a spurious peak in the profile.
            rotated = (
                mask
                if abs(angle) < 1e-6
                else cv2.warpAffine(
                    mask,
                    cv2.getRotationMatrix2D(
                        (mask.shape[1] / 2.0, mask.shape[0] / 2.0), angle, 1.0
                    ),
                    (mask.shape[1], mask.shape[0]),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
            )
            score = _profile_score(rotated)
            if score > best_s:
                best_a, best_s = float(angle), score
        return best_a, best_s

    coarse, _ = best_angle(np.arange(-max_angle, max_angle + coarse_step, coarse_step))
    fine, _ = best_angle(np.arange(coarse - coarse_step, coarse + coarse_step + 0.1, 0.1))

    # The estimator finds the rotation that *straightens* the image, which is
    # the negation of the skew present in it.
    skew = -fine
    return skew if abs(skew) <= max_angle else None


def estimate_skew_minarea(img: np.ndarray, *, max_angle: float = 20.0) -> float | None:
    """Estimate skew from the angles of individual text-line blobs.

    Characters are joined into line-shaped blobs with a wide, short kernel, and
    each blob's own `minAreaRect` angle is measured. The median across blobs is
    returned.

    Fitting one rectangle to *all* foreground pixels — the obvious
    implementation — does not work: it returns the bounding box of the entire
    text block, whose orientation is dominated by the block's overall aspect
    ratio rather than by the baselines, and on a roughly square block it snaps
    to 0 or 90 regardless of the true skew.
    """
    gray = to_gray(img)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    joined = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    angles: list[float] = []
    weights: list[float] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 120:
            continue

        (_, _), (bw, bh), angle = cv2.minAreaRect(contour)
        if min(bw, bh) < 3:
            continue

        # Normalise so the angle describes the blob's long axis.
        if bw < bh:
            angle += 90.0
        if angle > 45:
            angle -= 90
        elif angle < -45:
            angle += 90

        if abs(angle) <= max_angle:
            angles.append(float(angle))
            weights.append(float(area))

    if len(angles) < 2:
        return None

    order = np.argsort(angles)
    sorted_angles = np.asarray(angles)[order]
    cumulative = np.cumsum(np.asarray(weights)[order])
    median_idx = int(np.searchsorted(cumulative, cumulative[-1] / 2.0))
    return float(sorted_angles[min(median_idx, len(sorted_angles) - 1)])


def estimate_skew(img: np.ndarray, *, max_angle: float = 20.0) -> tuple[float, str]:
    """Combine the skew estimators. Returns (degrees, method).

    The projection profile is primary — it is the only one of the three that
    works on sparse text with no long straight lines, which is the normal case
    for a medicine strip. Hough and per-blob minAreaRect act as corroboration:
    when one agrees with the projection estimate within 2 degrees the two are
    averaged, which sharpens the result slightly. Disagreement is resolved in
    favour of the projection estimate rather than by averaging, since averaging
    a good estimate with a bad one just produces a mediocre one.
    """
    projection = estimate_skew_projection(img, max_angle=max_angle)
    hough = estimate_skew_hough(img, max_angle=max_angle)
    minarea = estimate_skew_minarea(img, max_angle=max_angle)

    if projection is not None:
        for other, name in ((hough, "hough"), (minarea, "minarea")):
            if other is not None and abs(other - projection) <= 2.0:
                return (projection + other) / 2.0, f"projection+{name}"
        return projection, "projection"

    if hough is not None:
        return hough, "hough"
    if minarea is not None:
        return minarea, "minarea"
    return 0.0, "none"


def rotate(img: np.ndarray, angle: float, *, expand: bool = True) -> np.ndarray:
    """Rotate about the centre, growing the canvas so nothing is clipped.

    Without `expand`, rotating a full-width strip pushes its ends outside the
    original frame and truncates exactly the text at the extremes — often the
    brand name.
    """
    if abs(angle) < 0.1:
        return img

    h, w = img.shape[:2]
    centre = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)

    if expand:
        cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        matrix[0, 2] += new_w / 2.0 - centre[0]
        matrix[1, 2] += new_h / 2.0 - centre[1]
        size = (new_w, new_h)
    else:
        size = (w, h)

    return cv2.warpAffine(
        img, matrix, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def deskew(img: np.ndarray, *, max_angle: float = 20.0) -> tuple[np.ndarray, dict]:
    """Estimate and correct residual in-plane rotation.

    `estimate_skew` reports the rotation *present in* the image, so the
    correction is its negation. Getting this sign wrong is silent and doubles
    the skew instead of removing it, which is why `test_deskew_round_trip`
    re-measures the output rather than trusting the estimate.
    """
    angle, method = estimate_skew(img, max_angle=max_angle)
    metrics = {"skew_deg": round(angle, 2), "skew_method": method}

    if abs(angle) < 0.3:
        metrics["skew_corrected"] = False
        return img, metrics

    metrics["skew_corrected"] = True
    return rotate(img, -angle), metrics


def apply(
    img: np.ndarray,
    quad: np.ndarray | None = None,
    *,
    do_rectify: bool = True,
    do_deskew: bool = True,
    max_skew_deg: float = 20.0,
) -> tuple[np.ndarray, dict]:
    """Perspective-rectify (if a quad was found) then deskew.

    Order matters: the projective warp removes the *out-of-plane* rotation, and
    only then is the remaining error a simple in-plane angle that a 2-D rotation
    can fix. Deskewing first would measure an angle that is not constant across
    the image, and "correcting" by it would make things worse.
    """
    metrics: dict = {"rectified": False}
    out = img

    if do_rectify and quad is not None:
        out = four_point_transform(out, quad)
        metrics["rectified"] = True

    if do_deskew:
        out, skew_metrics = deskew(out, max_angle=max_skew_deg)
        metrics.update(skew_metrics)

    return out, metrics
