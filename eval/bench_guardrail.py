#!/usr/bin/env python3
"""
Guardrail benchmark — what the contextual grounding filter actually scores.

The grounding threshold is the one guardrail setting with a real cost on both
sides. Too low and an explanation padded with the model's own pharmacological
knowledge survives. Too high and correct plain-English renderings of retrieved
facts are blocked, the explanation disappears from most scans, and the rational
response is to switch the guardrail off — which is worse than either.

So it gets measured rather than reasoned about. This script runs the REAL
system prompt against REAL fact sheets with trace enabled, and reports the
score distribution plus the pass rate at candidate thresholds.

Trace is enabled here and deliberately NOT in the application client: the trace
echoes back the text that triggered a filter, which in production could include
a user's own health details. This benchmark only ever sends catalogue drug
names, so there is nothing personal to leak.

    python -m eval.bench_guardrail
    python -m eval.bench_guardrail --drugs "Dolo 650 Tablet,Pan 40 Tablet"

Costs one Bedrock call per product on the fast model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from apps.api.config import get_settings  # noqa: E402
from apps.api.deps import AppState  # noqa: E402
from packages.reasoning.explainer import SYSTEM_PROMPT, render_fact_sheet  # noqa: E402

DEFAULT_DRUGS = [
    "Crocin Advance Tablet", "Augmentin 625 Duo Tablet", "Ecosprin 75 Tablet",
    "Dolo 650 Tablet", "Pan 40 Tablet", "Azithral 500 Tablet",
    "Montair LC Tablet", "Zerodol SP Tablet", "Telma 40 Tablet",
    "Glycomet 500 Tablet",
]

CANDIDATE_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)

QUESTION = "Explain this medicine in plain words for someone with no medical background."


def scores_for(client, settings, sheet: str) -> tuple[float | None, float | None, str]:
    """(grounding, relevance, stopReason) for one fact sheet."""
    response = client.converse(
        modelId=settings.bedrock_fast_model_id,
        system=[
            {"text": SYSTEM_PROMPT},
            {"guardContent": {"text": {"text": sheet, "qualifiers": ["grounding_source"]}}},
        ],
        messages=[
            {"role": "user", "content": [
                {"guardContent": {"text": {"text": QUESTION, "qualifiers": ["query"]}}}
            ]}
        ],
        inferenceConfig={"maxTokens": 600, "temperature": 0.0},
        guardrailConfig={
            "guardrailIdentifier": settings.bedrock_guardrail_id,
            "guardrailVersion": settings.bedrock_guardrail_version,
            "trace": "enabled",
        },
    )
    grounding = relevance = None
    assessments = response.get("trace", {}).get("guardrail", {}).get("outputAssessments", {})
    for assessment in assessments.values():
        for entry in (assessment if isinstance(assessment, list) else [assessment]):
            for f in entry.get("contextualGroundingPolicy", {}).get("filters", []):
                if f.get("type") == "GROUNDING":
                    grounding = f.get("score")
                elif f.get("type") == "RELEVANCE":
                    relevance = f.get("score")
    return grounding, relevance, response.get("stopReason", "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drugs", default=",".join(DEFAULT_DRUGS))
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "eval" / "results")
    args = parser.parse_args(argv)

    settings = get_settings()
    if not settings.bedrock_guardrail_id:
        print("error: BEDROCK_GUARDRAIL_ID is not set; nothing to measure.", file=sys.stderr)
        print("Create one with: python scripts/create_guardrail.py", file=sys.stderr)
        return 1

    import boto3

    credentials = {}
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        credentials = {
            "aws_access_key_id": settings.aws_access_key_id,
            "aws_secret_access_key": settings.aws_secret_access_key,
        }
    client = boto3.client("bedrock-runtime", region_name=settings.aws_region, **credentials)

    state = AppState()
    state.startup(settings)
    if not state.ready:
        print(f"error: not ready: {state.startup_errors}", file=sys.stderr)
        return 1

    rows = []
    for name in [d.strip() for d in args.drugs.split(",") if d.strip()]:
        result = state.orchestrator.analyse_text(name, explain=False)
        # Abstained identifications never reach the explainer, so scoring one
        # would measure a call the application does not make.
        if result.identification.status == "abstained":
            print(f"  {name:26} abstained — explainer never runs, skipped")
            continue
        grounding, relevance, stop = scores_for(client, settings, render_fact_sheet(result))
        rows.append({
            "drug": name,
            "identification": result.identification.status,
            "grounding": grounding,
            "relevance": relevance,
            "stop_reason": stop,
        })
        g = "  -  " if grounding is None else f"{grounding:.3f}"
        r = "  -  " if relevance is None else f"{relevance:.3f}"
        print(f"  {name:26} {result.identification.status:10} grounding={g} relevance={r}")

    values = sorted(r["grounding"] for r in rows if r["grounding"] is not None)
    if not values:
        print("\nno scores returned; is trace permitted for this guardrail?", file=sys.stderr)
        return 1

    print(f"\n  n={len(values)}  min={min(values):.2f}  "
          f"median={values[len(values) // 2]:.2f}  max={max(values):.2f}")
    print("\n  threshold   legitimate explanations passing")
    for t in CANDIDATE_THRESHOLDS:
        passing = sum(v >= t for v in values)
        bar = "#" * passing
        print(f"    {t:.1f}       {passing:2}/{len(values)}  {bar}")

    print("\n  Pick a threshold inside a GAP in the distribution, not on a cliff "
          "edge.\n  Scores: " + " ".join(f"{v:.2f}" for v in values))

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "guardrail.json"
    path.write_text(json.dumps({
        "guardrail_version": settings.bedrock_guardrail_version,
        "model": settings.bedrock_fast_model_id,
        "rows": rows,
        "pass_rate": {str(t): sum(v >= t for v in values) / len(values)
                      for t in CANDIDATE_THRESHOLDS},
    }, indent=2))
    print(f"\n  written {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
