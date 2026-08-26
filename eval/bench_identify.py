#!/usr/bin/env python3
"""
Identification benchmark — MediCure against a raw LLM, on identical inputs.

Measures four things, of which only the first is the one most systems report:

  accuracy          how often the top answer is right, answering everything
  coverage@95       fraction answered when held to 95% precision
  silent failure    how often it is confidently wrong — the number that matters
  calibration       whether stated confidence means anything (ECE)

The baseline arm sends the same corrupted queries to a Bedrock model with no
retrieval and asks it to name the composition. That is the honest comparison:
not "our RAG beats a model with no information", but "given the same damaged
OCR text, which produces a trustworthy answer".

The expected result is not that the LLM is inaccurate. It is that the LLM
cannot abstain. It will name a plausible Indian brand for a query matching
nothing, and its stated confidence is a token distribution rather than a
frequency — so its failures are invisible, and a patient cannot tell which
third of the answers to distrust.

    python -m eval.bench_identify --samples 300
    python -m eval.bench_identify --samples 300 --with-llm-baseline
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from packages.resolver.calibrate import (  # noqa: E402
    expected_calibration_error,
    load_or_default,
    risk_coverage,
)
from packages.resolver.corruption import CorruptionProfile, corrupt  # noqa: E402
from packages.resolver.index import DEFAULT_ARTIFACT_DIR, get_index  # noqa: E402


@dataclass
class ArmResult:
    """Scores for one system on one query set."""

    name: str
    n: int = 0
    correct: int = 0
    answered: int = 0
    abstained: int = 0
    wrong_and_confident: int = 0
    latency_ms: list[float] = field(default_factory=list)
    ece: float | None = None
    coverage: dict = field(default_factory=dict)
    by_severity: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        """Accuracy over everything, abstentions counted as wrong."""
        return self.correct / self.n if self.n else 0.0

    @property
    def precision_when_answering(self) -> float:
        return self.correct / self.answered if self.answered else 0.0

    @property
    def coverage_rate(self) -> float:
        return self.answered / self.n if self.n else 0.0

    @property
    def silent_failure_rate(self) -> float:
        """Confidently wrong, as a fraction of everything.

        The headline safety number. A system that abstains instead of guessing
        drives this toward zero without necessarily being more accurate.
        """
        return self.wrong_and_confident / self.n if self.n else 0.0

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("latency_ms")
        data.update(
            {
                "accuracy": round(self.accuracy, 4),
                "precision_when_answering": round(self.precision_when_answering, 4),
                "coverage_rate": round(self.coverage_rate, 4),
                "silent_failure_rate": round(self.silent_failure_rate, 4),
                "median_latency_ms": round(float(np.median(self.latency_ms)), 1)
                if self.latency_ms
                else None,
            }
        )
        return data


def build_queries(index, n_samples: int, seed: int) -> list[dict]:
    """Corrupted queries with known ground truth.

    Severity is stratified rather than pooled, because an aggregate number
    conceals the thing that matters: a system can look fine overall while being
    useless on exactly the damaged inputs it exists to handle.
    """
    rng = random.Random(seed)
    profiles = [
        ("light", CorruptionProfile.light()),
        ("moderate", CorruptionProfile.moderate()),
        ("heavy", CorruptionProfile.heavy()),
    ]

    rows = rng.sample(range(len(index)), min(n_samples * 2, len(index)))
    queries: list[dict] = []

    for row in rows:
        if len(queries) >= n_samples:
            break
        record = index.record(row)
        if not record.signature or not record.name:
            continue

        label, profile = profiles[len(queries) % 3]
        query = (
            f"{corrupt(record.name, profile, rng)} "
            f"{corrupt(record.composition, profile, rng)}"
        ).strip()
        if not query:
            continue

        queries.append(
            {
                "query": query,
                "truth_signature": record.signature,
                "truth_composition": record.composition,
                "truth_name": record.name,
                "severity": label,
            }
        )

    return queries


def run_medicure(index, calibrator, queries: list[dict]) -> ArmResult:
    arm = ArmResult(name="MediCure (retrieval + calibrated abstention)", n=len(queries))
    probabilities: list[float] = []
    correct_flags: list[int] = []
    severity: dict = {}

    for item in queries:
        started = time.perf_counter()
        matches = index.search_compositions(item["query"], top_k=5)
        status, probability = calibrator.decide(matches, item["query"])
        arm.latency_ms.append((time.perf_counter() - started) * 1000)

        is_correct = bool(matches) and matches[0].signature == item["truth_signature"]
        answered = status in {"confident", "ambiguous"}

        probabilities.append(probability)
        correct_flags.append(int(is_correct))

        if is_correct:
            arm.correct += 1
        if answered:
            arm.answered += 1
            if not is_correct and status == "confident":
                arm.wrong_and_confident += 1
        else:
            arm.abstained += 1

        bucket = severity.setdefault(item["severity"], {"n": 0, "correct": 0, "answered": 0})
        bucket["n"] += 1
        bucket["correct"] += int(is_correct)
        bucket["answered"] += int(answered)

    probabilities_array = np.asarray(probabilities)
    correct_array = np.asarray(correct_flags)

    ece, _ = expected_calibration_error(probabilities_array, correct_array)
    arm.ece = round(ece, 4)
    arm.coverage = risk_coverage(probabilities_array, correct_array)
    arm.by_severity = {
        k: {
            "n": v["n"],
            "accuracy": round(v["correct"] / v["n"], 4),
            "coverage": round(v["answered"] / v["n"], 4),
        }
        for k, v in sorted(severity.items())
    }

    if not calibrator.is_fitted:
        arm.notes.append(
            "calibrator not fitted — probabilities are raw similarities. "
            "Run scripts/fit_calibrator.py."
        )

    return arm


BASELINE_PROMPT = """You identify medicines from damaged OCR text of Indian medicine packaging.

Reply with ONLY a JSON object:
{"composition": "<active ingredients and strengths, e.g. 'paracetamol 500mg'>",
 "confidence": <0.0 to 1.0>}

If you cannot identify it, use an empty composition and a low confidence."""


def run_llm_baseline(queries: list[dict], settings) -> ArmResult:
    """The same queries against a Bedrock model with no retrieval.

    Deliberately given a fair chance: it is told the input is damaged OCR of
    Indian packaging, told the output shape, and explicitly permitted to
    decline. Whether it takes that option is the interesting part.
    """
    from packages.reasoning.bedrock import BedrockClient, BedrockUnavailable, parse_json_response

    arm = ArmResult(name="Raw LLM (no retrieval)", n=len(queries))

    try:
        client = BedrockClient(
            region=settings.aws_region,
            model_id=settings.bedrock_model_id,
            fast_model_id=settings.bedrock_fast_model_id,
            max_tokens=200,
            temperature=0.0,
        )
    except BedrockUnavailable as exc:
        arm.notes.append(f"baseline skipped: {exc}")
        return arm

    probabilities: list[float] = []
    correct_flags: list[int] = []

    for item in queries:
        started = time.perf_counter()
        try:
            response = client.converse(
                system=BASELINE_PROMPT,
                messages=[{"role": "user", "content": [{"text": item["query"]}]}],
                max_tokens=200,
            )
        except BedrockUnavailable as exc:
            arm.notes.append(f"call failed: {exc}")
            break
        arm.latency_ms.append((time.perf_counter() - started) * 1000)

        parsed = parse_json_response(response.text) or {}
        claimed = str(parsed.get("composition", "")).strip().lower()
        confidence = float(parsed.get("confidence") or 0.0)

        # Graded generously: any overlap of ingredient tokens with the truth
        # counts as correct. Exact signature matching would be unfair to a
        # system that was never given the vocabulary.
        truth_tokens = {
            t for t in item["truth_composition"].lower().replace("+", " ").split() if len(t) > 3
        }
        claimed_tokens = {t for t in claimed.replace("+", " ").split() if len(t) > 3}
        is_correct = bool(truth_tokens & claimed_tokens)

        answered = bool(claimed)
        probabilities.append(confidence)
        correct_flags.append(int(is_correct))

        if is_correct:
            arm.correct += 1
        if answered:
            arm.answered += 1
            if not is_correct and confidence >= 0.5:
                arm.wrong_and_confident += 1
        else:
            arm.abstained += 1

    if probabilities:
        ece, _ = expected_calibration_error(np.asarray(probabilities), np.asarray(correct_flags))
        arm.ece = round(ece, 4)
        arm.coverage = risk_coverage(np.asarray(probabilities), np.asarray(correct_flags))

    arm.n = len(correct_flags) or arm.n
    return arm


def render(arms: list[ArmResult]) -> str:
    lines = ["", "=" * 78, "IDENTIFICATION BENCHMARK", "=" * 78, ""]
    lines.append(f"{'metric':<34}" + "".join(f"{a.name[:20]:>22}" for a in arms))
    lines.append("-" * 78)

    def row(label: str, fn) -> None:
        lines.append(f"{label:<34}" + "".join(f"{fn(a):>22}" for a in arms))

    row("samples", lambda a: str(a.n))
    row("accuracy (answer everything)", lambda a: f"{a.accuracy:.1%}")
    row("answered (coverage)", lambda a: f"{a.coverage_rate:.1%}")
    row("precision when answering", lambda a: f"{a.precision_when_answering:.1%}")
    row("abstained", lambda a: f"{a.abstained / a.n:.1%}" if a.n else "-")
    row("SILENT FAILURE (conf. wrong)", lambda a: f"{a.silent_failure_rate:.1%}")
    row("calibration error (ECE)", lambda a: f"{a.ece:.3f}" if a.ece is not None else "-")
    row(
        "coverage @ 95% precision",
        lambda a: (
            f"{a.coverage.get('p95', {}).get('coverage', 0):.1%}"
            if a.coverage.get("p95", {}).get("achievable")
            else "unreachable"
        ),
    )
    row(
        "median latency",
        lambda a: f"{np.median(a.latency_ms):.0f} ms" if a.latency_ms else "-",
    )

    lines.append("")
    for arm in arms:
        if arm.by_severity:
            detail = "  ".join(
                f"{k}: {v['accuracy']:.0%} acc / {v['coverage']:.0%} answered"
                for k, v in arm.by_severity.items()
            )
            lines.append(f"{arm.name[:34]:<36}{detail}")
        for note in arm.notes:
            lines.append(f"  note: {note}")

    lines.append("")
    lines.append("Silent failure is the number that matters clinically: a wrong answer given")
    lines.append("confidently is one the user has no way to catch.")
    lines.append("=" * 78)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--with-llm-baseline", action="store_true",
                        help="also run the Bedrock no-retrieval arm (costs tokens)")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "eval" / "results")
    args = parser.parse_args(argv)

    try:
        index = get_index()
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    calibrator = load_or_default(DEFAULT_ARTIFACT_DIR)
    queries = build_queries(index, args.samples, args.seed)
    print(f"built {len(queries)} corrupted queries (stratified by severity)")

    arms = [run_medicure(index, calibrator, queries)]

    if args.with_llm_baseline:
        from apps.api.config import get_settings

        print("running LLM baseline (this costs Bedrock tokens)...")
        arms.append(run_llm_baseline(queries, get_settings()))

    print(render(arms))

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {"samples": len(queries), "seed": args.seed,
               "arms": [a.to_dict() for a in arms]}
    (args.out / "identify.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nwritten to {args.out / 'identify.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
