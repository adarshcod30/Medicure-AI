"""
Calibrated confidence and abstention — the project's headline differentiator.

A cosine similarity of 0.62 is not a probability. It is a number whose meaning
depends on how many competitors were close behind, how long the query was, and
how distinctive the match is. Reporting it to a user as "62% confident", or
thresholding on it directly, is the mistake this module exists to avoid.

What it produces instead is **P(the top composition is correct)**, fitted on
held-out data where the truth is known, so that among the answers scored 0.9,
about 90% really are right. That is a claim you can check, and
`eval/bench_identify.py` checks it.

Why this matters more than accuracy. A system answering every query at 81%
accuracy fails silently 19% of the time, and the user cannot tell which. A
system that answers 78% of queries at 95% precision and says "I am not sure"
on the rest fails *visibly*, and a visible failure on a medicine is one the
user can act on by asking a pharmacist. For a frontier LLM this is not
available at all: it will produce a fluent identification for a query that
matches nothing, and its stated confidence is a token distribution, not a
frequency.

Method: gradient-boosted trees over similarity features, then **isotonic
regression** to map the raw score onto probabilities. Isotonic is used rather
than Platt scaling because it assumes only monotonicity — higher score means
more likely correct — rather than a sigmoid shape the data need not have.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

from .corruption import CorruptionProfile, corrupt
from .index import DEFAULT_ARTIFACT_DIR, BrandIndex, CompositionMatch

CALIBRATOR_VERSION = 1

MIN_LEXICAL_SUPPORT = 3
"""Alphabetic tokens of 4+ characters required before answering *confidently*.

Three rather than one: a single real word alongside a lucky numeric match is
still thin evidence, and OCR emits plausible-looking non-words ("tens", "tthe")
often enough that one is not a meaningful bar. Below the threshold the answer
is downgraded to `ambiguous` rather than discarded — the candidate may well be
right, and the user still sees it, but it is not presented as settled."""

FEATURE_NAMES = (
    # --- match quality: how good is the best candidate ---
    "top_similarity",
    "margin",
    "margin_ratio",
    "second_similarity",
    "support",
    "candidate_count",
    # --- query quality: is the question even answerable ---
    "query_length",
    "query_tokens",
    "mean_token_length",
    "long_token_fraction",
    "longest_token",
    "has_digits",
)


def extract_features(matches: list[CompositionMatch], query: str) -> np.ndarray:
    """Turn a ranked candidate list and its query into a feature vector.

    Two families, and the second was added only after real images exposed why
    it was needed.

    **Match quality.** `margin` — the gap between best and second-best — is the
    most informative of these and the one a bare similarity threshold throws
    away. A top score of 0.62 with the runner-up at 0.61 means the index found
    two equally good answers and cannot choose; the same 0.62 with the runner-up
    at 0.20 is a clean isolated match. Identical similarity, opposite meaning.

    **Query quality.** Originally absent, and its absence was a real failure. On
    a 242x208 retail thumbnail, OCR returned `['ae', 'wey', 'be', 'tablets']` —
    the composition print is a few pixels tall and unreadable. One composition
    happened to match those fragments well and with a wide margin, which is
    exactly the pattern that meant "correct" in training, so the calibrator
    returned **P = 0.93, confident, and wrong**.

    Nothing in the match-quality features can catch that, because the match
    genuinely was clean — of a degenerate query. The training distribution never
    contained one: synthetic corruption always derives from a real product name,
    so it never produces two-letter noise. Real OCR on an unreadable image
    produces almost nothing else.

    `mean_token_length` and `long_token_fraction` capture it directly. A query
    whose tokens average three characters carries no identifying information no
    matter how well something matches it.
    """
    tokens = [t for t in query.split() if t]
    lengths = [len(t) for t in tokens]

    query_features = [
        float(len(query)),
        float(len(tokens)),
        float(np.mean(lengths)) if lengths else 0.0,
        float(sum(1 for n in lengths if n >= 4) / len(lengths)) if lengths else 0.0,
        float(max(lengths)) if lengths else 0.0,
        float(any(c.isdigit() for c in query)),
    ]

    if not matches:
        return np.array([0.0] * 6 + query_features, dtype=np.float32)

    top = matches[0].top_similarity
    second = matches[1].top_similarity if len(matches) > 1 else 0.0

    return np.array(
        [
            top,
            top - second,
            (top - second) / top if top > 0 else 0.0,
            second,
            float(matches[0].support),
            float(len(matches)),
            *query_features,
        ],
        dtype=np.float32,
    )


@dataclass
class CalibrationReport:
    """How well the calibrator performs, on held-out data."""

    n_samples: int
    accuracy: float
    expected_calibration_error: float
    brier_score: float
    coverage_at_precision: dict[str, dict] = field(default_factory=dict)
    reliability_bins: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "accuracy": round(self.accuracy, 4),
            "expected_calibration_error": round(self.expected_calibration_error, 4),
            "brier_score": round(self.brier_score, 4),
            "coverage_at_precision": self.coverage_at_precision,
            "reliability_bins": self.reliability_bins,
        }


def expected_calibration_error(
    probabilities: np.ndarray, correct: np.ndarray, bins: int = 10
) -> tuple[float, list[dict]]:
    """ECE, plus the reliability diagram it is computed from.

    Buckets predictions by confidence and compares each bucket's mean
    confidence to its actual accuracy. A perfectly calibrated model has
    predictions of 0.7 correct 70% of the time; ECE is the average gap,
    weighted by bucket size.

    The bins are returned as well, because a single ECE number hides *where*
    the miscalibration sits — and over-confidence at the top of the range is
    far more dangerous here than under-confidence at the bottom.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(probabilities)
    error = 0.0
    diagram: list[dict] = []

    for i in range(bins):
        low, high = edges[i], edges[i + 1]
        mask = (probabilities > low) & (probabilities <= high) if i > 0 else (
            probabilities >= low
        ) & (probabilities <= high)
        count = int(mask.sum())
        if count == 0:
            continue

        confidence = float(probabilities[mask].mean())
        accuracy = float(correct[mask].mean())
        error += (count / total) * abs(confidence - accuracy)

        diagram.append(
            {
                "bin": f"{low:.1f}-{high:.1f}",
                "count": count,
                "mean_confidence": round(confidence, 4),
                "accuracy": round(accuracy, 4),
                "gap": round(confidence - accuracy, 4),
            }
        )

    return error, diagram


def risk_coverage(
    probabilities: np.ndarray, correct: np.ndarray, targets: tuple[float, ...] = (0.90, 0.95, 0.99)
) -> dict[str, dict]:
    """Coverage achievable at each target precision.

    This is the table the whole abstention argument rests on: "at 95%
    precision the system answers X% of queries". Computed by sweeping the
    threshold and finding the lowest one that still meets the target — the
    lowest, because that maximises how many queries get answered.
    """
    order = np.argsort(-probabilities)
    sorted_correct = correct[order]

    cumulative_correct = np.cumsum(sorted_correct)
    counts = np.arange(1, len(sorted_correct) + 1)
    precision_curve = cumulative_correct / counts

    results: dict[str, dict] = {}
    for target in targets:
        feasible = np.where(precision_curve >= target)[0]
        if len(feasible) == 0:
            results[f"p{int(target * 100)}"] = {
                "achievable": False,
                "coverage": 0.0,
                "threshold": 1.0,
                "note": "target precision not reachable at any threshold",
            }
            continue

        cut = int(feasible[-1])
        results[f"p{int(target * 100)}"] = {
            "achievable": True,
            "coverage": round((cut + 1) / len(sorted_correct), 4),
            "precision": round(float(precision_curve[cut]), 4),
            "threshold": round(float(probabilities[order][cut]), 4),
            "answered": cut + 1,
            "total": len(sorted_correct),
        }

    return results


class Calibrator:
    """Maps candidate-list features onto a calibrated P(correct)."""

    def __init__(self, model=None, isotonic=None, threshold: float = 0.5,
                 report: CalibrationReport | None = None):
        self.model = model
        self.isotonic = isotonic
        self.threshold = threshold
        self.report = report

    @property
    def is_fitted(self) -> bool:
        return self.model is not None and self.isotonic is not None

    def probability(self, matches: list[CompositionMatch], query: str) -> float:
        """P(top candidate is the correct composition).

        Falls back to the raw similarity when no calibrator has been fitted.
        The fallback is deliberately conservative and clearly labelled by
        `is_fitted` — an uncalibrated score must never be presented as a
        probability, since that is the exact failure this module addresses.
        """
        if not matches:
            return 0.0
        if not self.is_fitted:
            return float(min(matches[0].top_similarity, 0.99))

        features = extract_features(matches, query).reshape(1, -1)
        raw = float(self.model.predict_proba(features)[0, 1])
        return float(np.clip(self.isotonic.predict([raw])[0], 0.0, 1.0))

    @staticmethod
    def lexical_support(query: str) -> int:
        """Count word-like tokens: alphabetic, four characters or more.

        Numbers and short fragments do not count. See `decide` for why.
        """
        return sum(1 for t in query.split() if len(t) >= 4 and t.isalpha())

    def decide(self, matches: list[CompositionMatch], query: str) -> tuple[str, float]:
        """Return (status, probability): 'confident', 'ambiguous' or 'abstained'.

        Three states rather than two. 'ambiguous' means the system has real
        candidates but cannot choose between them — which is a materially
        different message to the user than "I could not read this", and calls
        for a different action: confirm which of these two it is, rather than
        retake the photo.
        """
        probability = self.probability(matches, query)

        # A hard gate, independent of the learned probability: an
        # identification resting on numbers alone is not an identification.
        #
        # Measured failure. On a low-resolution product shot, OCR returned
        # ['x0.035mg', 'x0.04mg', 'x0.03mg', 'ree', 'ore', 'tens', 'tthe'].
        # 0.035mg is the exact strength of ethinyl estradiol — distinctive
        # enough to produce a strong, wide-margin match — and the calibrator
        # scored it 0.575 confident and wrong. The query-quality features do not
        # catch it because `x0.035mg` is eight characters, so mean token length
        # looks healthy. The problem is not token length; it is that no WORD
        # supported the match.
        #
        # Defence in depth rather than a replacement for calibration: the
        # learned features handle degenerate queries in general, this handles
        # the specific case where a numeral carries the whole match.
        if probability >= self.threshold:
            if self.lexical_support(query) < MIN_LEXICAL_SUPPORT:
                return "ambiguous", min(probability, self.threshold * 0.9)
            return "confident", probability
        if matches and probability >= self.threshold * 0.5:
            return "ambiguous", probability
        return "abstained", probability

    def save(self, path: Path) -> None:
        joblib.dump(
            {
                "version": CALIBRATOR_VERSION,
                "model": self.model,
                "isotonic": self.isotonic,
                "threshold": self.threshold,
                "report": self.report.to_dict() if self.report else None,
            },
            path,
            compress=3,
        )

    @classmethod
    def load(cls, path: Path) -> Calibrator:
        payload = joblib.load(path)
        if payload.get("version") != CALIBRATOR_VERSION:
            raise RuntimeError(f"calibrator version mismatch in {path}")
        return cls(payload["model"], payload["isotonic"], payload["threshold"])

    @classmethod
    def unfitted(cls) -> Calibrator:
        """A pass-through calibrator, for before `scripts/fit_calibrator.py` runs."""
        return cls()


def build_training_data(
    index: BrandIndex,
    *,
    n_products: int = 3000,
    seed: int = 0,
    profiles: tuple[CorruptionProfile, ...] = (),
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate labelled (features, correct) pairs from synthetic corruptions.

    Each sample is a query built by corrupting a real product's name and
    composition, searched against the index, and labelled by whether the
    top-ranked *composition* is the true one.

    The three severity profiles are sampled evenly on purpose. Training only on
    easy queries produces a calibrator that has never seen a genuinely
    unanswerable input and therefore never learns to output a low probability —
    which would defeat the entire point.
    """
    profiles = profiles or (
        CorruptionProfile.light(),
        CorruptionProfile.moderate(),
        CorruptionProfile.heavy(),
    )

    rng = random.Random(seed)
    rows = rng.sample(range(len(index)), min(n_products, len(index)))

    features: list[np.ndarray] = []
    labels: list[int] = []
    queries: list[str] = []

    for row in rows:
        record = index.record(row)
        if not record.signature:
            continue

        # An uncorrupted query, alongside the damaged ones. Training only on
        # corrupted input teaches the calibrator that clean input does not
        # exist: typing an exact brand name into the search box was scoring
        # P=0.69 and landing in "ambiguous", because nothing in the training
        # distribution ever matched that well. Users type names correctly far
        # more often than OCR reads them correctly.
        clean_queries = [record.name, f"{record.name} {record.composition}".strip()]

        for query in clean_queries:
            if not query.strip():
                continue
            matches = index.search_compositions(query, top_k=5)
            features.append(extract_features(matches, query))
            labels.append(int(bool(matches) and matches[0].signature == record.signature))
            queries.append(query)

        for profile in profiles:
            name_part = corrupt(record.name, profile, rng)
            composition_part = corrupt(record.composition, profile, rng)
            query = f"{name_part} {composition_part}".strip()
            if not query:
                continue

            matches = index.search_compositions(query, top_k=5)
            features.append(extract_features(matches, query))
            labels.append(int(bool(matches) and matches[0].signature == record.signature))
            queries.append(query)

    # --- negative control 1: nonsense words ---
    # Without these the calibrator only ever sees inputs that have a correct
    # answer somewhere in the index, and learns that high similarity always
    # means correct. Real users photograph cosmetics, food packets and
    # handwriting.
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    for _ in range(len(rows) // 4):
        nonsense = " ".join(
            "".join(rng.choice(alphabet) for _ in range(rng.randint(4, 12)))
            for _ in range(rng.randint(1, 3))
        )
        matches = index.search_compositions(nonsense, top_k=5)
        features.append(extract_features(matches, nonsense))
        labels.append(0)
        queries.append(nonsense)

    # --- negative control 2: degenerate OCR fragments ---
    #
    # This class exists because of a measured failure, not a hypothetical one.
    # On low-resolution retail thumbnails the composition print is a few pixels
    # tall and Tesseract returns short fragments: `['ae', 'wey', 'be',
    # 'tablets']`, `['sa', 'ae']`, `['wi']`. One composition matched those
    # cleanly and with a wide margin, and the calibrator — which had never seen
    # such a query — scored it 0.93 and got it wrong.
    #
    # Corruption of a real name cannot generate this: it always preserves
    # enough structure to stay word-like. So the failure mode has to be
    # synthesised explicitly, sampled from the character bigrams OCR actually
    # emits on unreadable input.
    fragments = [
        "ae", "wey", "be", "ss", "es", "as", "sy", "oo", "zz", "se", "za", "lz",
        "we", "lee", "ay", "ie", "hy", "we", "lr", "gy", "wi", "sa", "at", "ty",
        "re", "ye", "ey", "ai", "ne", "ni", "ii", "ig", "go", "bs", "dag", "oe",
    ]
    filler = ["tablets", "tablet", "ltd", "mg", "india", "capsules", "strip"]

    for _ in range(len(rows) // 3):
        # Mostly noise, occasionally with one real-looking word — which is what
        # OCR does when a single large glyph survives on an unreadable strip.
        parts = [rng.choice(fragments) for _ in range(rng.randint(1, 6))]
        if rng.random() < 0.4:
            parts.append(rng.choice(filler))
        degenerate = " ".join(parts)

        matches = index.search_compositions(degenerate, top_k=5)
        features.append(extract_features(matches, degenerate))
        labels.append(0)
        queries.append(degenerate)

    return np.vstack(features), np.asarray(labels), queries


def fit(
    index: BrandIndex,
    *,
    n_products: int = 3000,
    target_precision: float = 0.95,
    seed: int = 0,
    verbose: bool = True,
) -> Calibrator:
    """Fit a calibrator and choose an abstention threshold.

    Training and calibration use disjoint splits. Fitting isotonic regression on
    the same predictions the classifier was trained on would map an
    over-confident model onto its own over-confidence and report near-perfect
    calibration — the numbers would look excellent and mean nothing.
    """

    def log(message: str) -> None:
        if verbose:
            print(f"[calibrate] {message}", flush=True)

    log(f"generating training data from {n_products} products...")
    features, labels, _ = build_training_data(index, n_products=n_products, seed=seed)
    log(f"  {len(labels):,} samples, {labels.mean():.1%} correct")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(labels))
    n_train = int(len(order) * 0.6)
    n_calib = int(len(order) * 0.2)

    train_idx = order[:n_train]
    calib_idx = order[n_train : n_train + n_calib]
    test_idx = order[n_train + n_calib :]

    log(f"  split: {len(train_idx)} train / {len(calib_idx)} calibrate / {len(test_idx)} test")

    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.9, random_state=seed
    )
    model.fit(features[train_idx], labels[train_idx])

    calib_raw = model.predict_proba(features[calib_idx])[:, 1]
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic.fit(calib_raw, labels[calib_idx])

    test_raw = model.predict_proba(features[test_idx])[:, 1]
    test_probabilities = np.clip(isotonic.predict(test_raw), 0.0, 1.0)
    test_correct = labels[test_idx]

    ece, diagram = expected_calibration_error(test_probabilities, test_correct)
    coverage = risk_coverage(test_probabilities, test_correct)
    brier = float(np.mean((test_probabilities - test_correct) ** 2))

    report = CalibrationReport(
        n_samples=len(test_idx),
        accuracy=float(test_correct.mean()),
        expected_calibration_error=ece,
        brier_score=brier,
        coverage_at_precision=coverage,
        reliability_bins=diagram,
    )

    key = f"p{int(target_precision * 100)}"
    entry = coverage.get(key, {})
    threshold = float(entry.get("threshold", 0.5)) if entry.get("achievable") else 0.5

    log(f"  accuracy (answer everything) : {report.accuracy:.3f}")
    log(f"  ECE                          : {ece:.4f}")
    log(f"  Brier                        : {brier:.4f}")
    for name, value in coverage.items():
        if value.get("achievable"):
            log(f"  at {name} precision: coverage {value['coverage']:.1%}, "
                f"threshold {value['threshold']:.3f}")
        else:
            log(f"  at {name} precision: NOT ACHIEVABLE")
    log(f"  abstention threshold set to {threshold:.3f}")

    return Calibrator(model, isotonic, threshold, report)


def load_or_default(artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> Calibrator:
    """Load the fitted calibrator, or a pass-through one if it is absent."""
    path = Path(artifact_dir) / "calibrator.joblib"
    if not path.exists():
        return Calibrator.unfitted()
    try:
        return Calibrator.load(path)
    except Exception:  # noqa: BLE001 - a stale artifact must not block startup
        return Calibrator.unfitted()
