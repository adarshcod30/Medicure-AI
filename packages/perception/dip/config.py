"""
DIP pipeline configuration.

Every processing stage is an independent switch. This exists so that
`eval/bench_ocr.py` can ablate the pipeline one stage at a time and produce a
table showing what each technique actually contributes to OCR accuracy — rather
than asserting that the preprocessing helps.

`DipConfig.raw()` is the ablation baseline: no processing at all, straight to OCR.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

DenoiseMethod = Literal["auto", "gaussian", "median", "bilateral", "nlm", "none"]
IllumMethod = Literal["homomorphic", "retinex", "divide", "none"]
EdgeMethod = Literal["canny", "sobel", "scharr", "log", "morph_gradient"]
BinarizeMethod = Literal["otsu", "adaptive_mean", "adaptive_gaussian", "sauvola", "niblack", "wolf"]


@dataclass(frozen=True)
class DipConfig:
    """A fully-specified DIP pipeline. Frozen so a config can be a cache key."""

    # ---- acquisition -----------------------------------------------------
    max_dimension: int = 2000
    """Longest edge, in px. Downscaling first keeps NLM denoising tractable."""

    auto_orient: bool = True
    """Honour the EXIF orientation tag. Phone photos are routinely 90 degrees off."""

    # ---- noise -----------------------------------------------------------
    denoise: bool = True
    denoise_method: DenoiseMethod = "auto"
    """'auto' estimates the noise level and picks: NLM (heavy), bilateral
    (moderate, edge-preserving), median (impulse), or skips entirely (clean)."""

    # ---- illumination and specular glare ---------------------------------
    remove_glare: bool = True
    """Detect blown-out specular highlights and inpaint them. The single most
    important stage for foil blister packs."""

    glare_v_threshold: int = 245
    """HSV Value above which a pixel is a specular-highlight candidate."""

    glare_s_threshold: int = 40
    """HSV Saturation below which that candidate is confirmed. Specular
    reflections are bright AND desaturated — that conjunction is what separates
    them from legitimately bright white packaging."""

    glare_inpaint_method: Literal["telea", "ns"] = "telea"

    normalize_illumination: bool = True
    illum_method: IllumMethod = "divide"
    """'divide' (image / heavily-blurred image) is fast and very effective for
    the smooth lighting gradients typical of a handheld phone shot.
    'homomorphic' separates illumination from reflectance in the log-frequency
    domain — stronger, slower. 'retinex' is multi-scale Retinex."""

    # ---- contrast --------------------------------------------------------
    clahe: bool = True
    clahe_clip_limit: float = 3.0
    clahe_tile_grid: int = 8
    """Contrast-Limited Adaptive Histogram Equalization. Applied to the L
    channel in LAB space so colour is not distorted. The clip limit is what
    stops it amplifying noise into texture."""

    gamma: float | None = None
    """None = skip. <1 brightens shadows, >1 darkens highlights."""

    unsharp: bool = True
    unsharp_amount: float = 1.0
    unsharp_radius: int = 5

    # ---- geometry --------------------------------------------------------
    detect_boundary: bool = True
    """Find the packet/strip quadrilateral so background is excluded."""

    rectify: bool = True
    """4-point perspective warp onto a fronto-parallel plane. Recovers text
    geometry that no downstream model can recover on its own."""

    deskew: bool = True
    """Residual in-plane rotation via Hough lines / minAreaRect."""

    max_skew_correction_deg: float = 20.0
    """Refuse to 'correct' beyond this — a large estimate usually means the
    estimator locked onto packaging graphics rather than a text baseline."""

    # ---- edges (used by boundary detection, and benchmarked directly) -----
    edge_method: EdgeMethod = "canny"
    canny_sigma: float = 0.33
    """Auto-hysteresis: lower = max(0, (1-sigma)*median), upper = min(255,
    (1+sigma)*median). Deriving thresholds from the image median instead of
    hardcoding (50, 150) is what keeps boundary detection working across the
    glare/blur/exposure variation in a real photo set."""

    # ---- morphology ------------------------------------------------------
    tophat: bool = True
    tophat_kernel: int = 15
    """White top-hat = image minus its opening. Isolates small bright
    structures against a varying background — which is exactly what embossed or
    debossed text on foil is. Plain Otsu on the same image returns noise."""

    blackhat: bool = False
    """The dark-text-on-bright-packaging counterpart. Off by default; the
    rendition fan-out tries it anyway."""

    # ---- periodic noise --------------------------------------------------
    notch_filter: bool = False
    """FFT notch filtering for moire/periodic artefacts. Off by default: it is
    expensive and only matters for scanned or screen-photographed input."""

    # ---- binarization ----------------------------------------------------
    binarize_methods: tuple[BinarizeMethod, ...] = ("sauvola", "otsu", "adaptive_gaussian")
    """Multiple binarizations are produced and all are OCR'd. Sauvola is the
    correct default for unevenly-lit packaging (it thresholds on a local mean
    and standard deviation); Otsu is a global method and fails under a lighting
    gradient, but wins on clean flat labels — so we run both and fuse."""

    sauvola_window: int = 25
    sauvola_k: float = 0.2
    niblack_k: float = -0.2

    # ---- rendition fan-out -----------------------------------------------
    rotations: tuple[int, ...] = (0, 90, 270)
    """Indian blister strips very often print the composition vertically along
    the edge. Tesseract will not find it without the rotated passes."""

    max_renditions: int = 9
    """Hard cap on (binarization x rotation) combinations, for latency."""

    # ---- text region detection -------------------------------------------
    text_detection: bool = True
    text_method: Literal["mser", "swt", "none"] = "mser"

    # ---- debugging -------------------------------------------------------
    dump_stages: bool = False
    """When set, pipeline.run() retains every intermediate array for the DIP
    inspector panel and for `--dump-stages`."""

    stage_order: tuple[str, ...] = field(
        default=(
            "acquire",
            "denoise",
            "glare",
            "illumination",
            "boundary",
            "rectify",
            "deskew",
            "clahe",
            "morphology",
            "sharpen",
            "binarize",
        )
    )

    # ---- presets ---------------------------------------------------------

    @classmethod
    def raw(cls) -> DipConfig:
        """The ablation baseline: decode the image and hand it straight to OCR.

        This is the 'Raw image -> Tesseract' row of the DIP results table. Every
        other row is this plus one more stage.
        """
        return cls(
            denoise=False,
            remove_glare=False,
            normalize_illumination=False,
            clahe=False,
            unsharp=False,
            detect_boundary=False,
            rectify=False,
            deskew=False,
            tophat=False,
            text_detection=False,
            binarize_methods=(),
            rotations=(0,),
        )

    @classmethod
    def fast(cls) -> DipConfig:
        """Latency-first: skip the expensive stages (NLM, boundary, rotations)."""
        return cls(
            denoise_method="bilateral",
            normalize_illumination=False,
            detect_boundary=False,
            rectify=False,
            binarize_methods=("sauvola",),
            rotations=(0,),
            max_renditions=2,
            text_detection=False,
        )

    @classmethod
    def full(cls) -> DipConfig:
        """Everything on. The default for a user-submitted scan."""
        return cls()

    @classmethod
    def foil(cls) -> DipConfig:
        """Tuned for reflective foil blister packs with embossed text."""
        return cls(
            denoise_method="bilateral",
            remove_glare=True,
            illum_method="homomorphic",
            clahe_clip_limit=4.0,
            tophat=True,
            tophat_kernel=21,
            binarize_methods=("sauvola", "niblack", "adaptive_gaussian"),
        )

    def with_(self, **kwargs) -> DipConfig:
        """Return a copy with fields overridden. Used to build ablation ladders."""
        return replace(self, **kwargs)


# The ablation ladder used by eval/bench_ocr.py. Each entry adds exactly one
# capability to the previous, so the delta in CER is attributable to that stage.
ABLATION_LADDER: list[tuple[str, DipConfig]] = [
    ("raw", DipConfig.raw()),
    ("+denoise", DipConfig.raw().with_(denoise=True)),
    ("+clahe", DipConfig.raw().with_(denoise=True, clahe=True)),
    ("+glare_inpaint", DipConfig.raw().with_(denoise=True, clahe=True, remove_glare=True)),
    (
        "+illumination",
        DipConfig.raw().with_(
            denoise=True, clahe=True, remove_glare=True, normalize_illumination=True
        ),
    ),
    (
        "+rectify",
        DipConfig.raw().with_(
            denoise=True,
            clahe=True,
            remove_glare=True,
            normalize_illumination=True,
            detect_boundary=True,
            rectify=True,
            deskew=True,
        ),
    ),
    (
        "+tophat",
        DipConfig.raw().with_(
            denoise=True,
            clahe=True,
            remove_glare=True,
            normalize_illumination=True,
            detect_boundary=True,
            rectify=True,
            deskew=True,
            tophat=True,
        ),
    ),
    ("+sauvola", DipConfig().with_(binarize_methods=("sauvola",), rotations=(0,))),
    ("+otsu_only", DipConfig().with_(binarize_methods=("otsu",), rotations=(0,))),
    ("full_fusion", DipConfig.full()),
]
