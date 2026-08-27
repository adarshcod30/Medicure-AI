#!/usr/bin/env python3
"""
Create the Bedrock guardrail that enforces groundedness mechanically.

    python scripts/create_guardrail.py

Prints a guardrail ID to put in `.env` as `BEDROCK_GUARDRAIL_ID`.

Why this exists. The explainer already has three defences, and they are not
equally strong:

  1. Structural — the model writes only into `explanation.text`, so a
     hallucinated price cannot reach `price_check.overcharge_percent`.
  2. Prompt — "use only what is in the fact sheet".
  3. Guardrail — this file.

Without (3), groundedness rests on (2), and a prompt is a *request*. The system
this project replaces had a prompt too; it said "add approximate Indian price
inside brackets", and got exactly that. The contextual grounding filter scores
every generated sentence against the retrieved fact sheet and blocks the
response below threshold, independently of whether the model felt like
complying.

The denied topics matter as much. This system explains what a medicine IS. It
must never tell someone what to take, how much, or whether it is safe for them
— and those are precisely the questions a helpful-sounding model drifts toward.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

GROUNDING_THRESHOLD = 0.50
"""How well the answer must be supported by the fact sheet to survive.

0.70 was the reasoned choice and it was too strict. Measured against the real
SYSTEM_PROMPT and real fact sheets for 10 catalogue products (trace enabled,
Nova Lite, maxTokens 600), the grounding scores were:

    0.88  0.80  0.78  0.70  0.68  0.65  0.52  |  0.28  0.27  0.17

    threshold   legitimate explanations passing
      0.70                4/10
      0.60                6/10
      0.50                7/10
      0.40                7/10

At 0.70 the filter blocked 5 of 7 attempted explanations, including a
*confident* paracetamol identification — the exact "teaches nothing except to
turn the guardrail off" failure this constant's previous docstring warned
about, arrived at from the other direction.

0.50 sits inside the natural gap between 0.52 and 0.28. Below it there is no
further gain (0.40 passes the same 7), and above it correct plain-English
renderings start dying. The three that still fail — 0.28, 0.27, 0.17 — are
outputs where the model padded a thin multi-ingredient fact sheet with its own
pharmacological knowledge, which is precisely what should be blocked.

The tension is real and worth naming: this explainer is *asked* to paraphrase
("painkiller" not "analgesic"), and paraphrase scores lower than quotation on
contextual grounding. A stricter threshold does not buy more safety here, it
buys fewer answers, because every load-bearing fact is already deterministic
and the model cannot reach any of them.

Re-measure with: python -m eval.bench_guardrail"""

RELEVANCE_THRESHOLD = 0.70
"""Left at 0.70. Across the same 10 products relevance scored 1.00 every time
-- it has never been the binding constraint, so there is nothing to loosen."""

DENIED_TOPICS = [
    {
        "name": "MedicalDiagnosis",
        "definition": (
            "Diagnosing a condition, interpreting symptoms, or telling a person what "
            "illness they have or may have."
        ),
        "examples": [
            "Based on your symptoms you probably have a bacterial infection.",
            "This sounds like it could be an ulcer.",
            "You should take this because you have a fever.",
        ],
        "type": "DENY",
    },
    {
        "name": "DosageAdvice",
        "definition": (
            "Recommending how much of a medicine to take, how often, for how long, or "
            "advising a person to start, stop, increase or decrease a dose."
        ),
        "examples": [
            "Take two tablets twice a day after meals.",
            "You can safely double the dose if the pain continues.",
            "Stop taking this once you feel better.",
        ],
        "type": "DENY",
    },
    {
        "name": "SafetyAssurance",
        "definition": (
            "Asserting that a medicine is safe for a person, safe in pregnancy, safe "
            "with alcohol, or safe alongside another medicine."
        ),
        "examples": [
            "This is completely safe to take during pregnancy.",
            "It is fine to drink alcohol with this medicine.",
            "There is no problem taking this with your blood pressure tablets.",
        ],
        "type": "DENY",
    },
]


def build_config(name: str) -> dict:
    return {
        "name": name,
        "description": (
            "Enforces that MediCure explanations stay grounded in retrieved facts, "
            "and blocks diagnosis, dosage advice and safety assurances."
        ),
        "contextualGroundingPolicyConfig": {
            "filtersConfig": [
                {"type": "GROUNDING", "threshold": GROUNDING_THRESHOLD},
                {"type": "RELEVANCE", "threshold": RELEVANCE_THRESHOLD},
            ]
        },
        "topicPolicyConfig": {"topicsConfig": DENIED_TOPICS},
        "contentPolicyConfig": {
            "filtersConfig": [
                {"type": "SEXUAL", "inputStrength": "HIGH", "outputStrength": "HIGH"},
                {"type": "VIOLENCE", "inputStrength": "HIGH", "outputStrength": "HIGH"},
                {"type": "HATE", "inputStrength": "HIGH", "outputStrength": "HIGH"},
                {"type": "INSULTS", "inputStrength": "HIGH", "outputStrength": "HIGH"},
                {"type": "MISCONDUCT", "inputStrength": "HIGH", "outputStrength": "HIGH"},
                {"type": "PROMPT_ATTACK", "inputStrength": "HIGH", "outputStrength": "NONE"},
            ]
        },
        # No PII policy. Bedrock masks PII in the API response but still writes
        # the ORIGINAL unmasked text to CloudWatch Logs, so enabling it here
        # would create a false sense of protection for a medical application.
        # Nothing in this pipeline sends user PII to the model anyway — the
        # fact sheet is retrieved database records.
        "blockedInputMessaging": (
            "MediCure can explain what a medicine is, but cannot give medical advice. "
            "Please ask a pharmacist or doctor."
        ),
        "blockedOutputsMessaging": (
            "That answer could not be verified against the retrieved records, so it "
            "was withheld. The facts shown above were retrieved and computed directly."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="medicure-grounding")
    parser.add_argument("--dry-run", action="store_true", help="print the config, create nothing")
    parser.add_argument(
        "--update",
        action="store_true",
        help="apply this config to an existing guardrail of the same name "
             "(needed after changing a threshold; the ID does not change)",
    )
    args = parser.parse_args(argv)

    config = build_config(args.name)

    if args.dry_run:
        print(json.dumps(config, indent=2))
        return 0

    try:
        import boto3
    except ImportError:
        print("error: boto3 not installed", file=sys.stderr)
        return 1

    from apps.api.config import get_settings

    settings = get_settings()
    credentials = {}
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        credentials = {
            "aws_access_key_id": settings.aws_access_key_id,
            "aws_secret_access_key": settings.aws_secret_access_key,
        }

    client = boto3.client("bedrock", region_name=settings.aws_region, **credentials)

    existing = None
    try:
        for guardrail in client.list_guardrails().get("guardrails", []):
            if guardrail.get("name") == args.name:
                existing = guardrail
                break
    except Exception as exc:  # noqa: BLE001
        print(f"error: could not list guardrails: {exc}", file=sys.stderr)
        print("The IAM policy needs bedrock:ListGuardrails and bedrock:CreateGuardrail.",
              file=sys.stderr)
        return 1

    if existing and not args.update:
        print(f"guardrail '{args.name}' already exists: {existing['id']}")
        print("Re-run with --update to apply the current thresholds to it.")
        print(f"\nBEDROCK_GUARDRAIL_ID={existing['id']}")
        return 0

    if existing:
        # Update in place so the ID in .env stays valid. A new guardrail would
        # mean a new ID and a silently unenforced deployment until someone
        # noticed .env still pointed at the old one.
        try:
            client.update_guardrail(guardrailIdentifier=existing["id"], **config)
        except Exception as exc:  # noqa: BLE001
            print(f"error: update_guardrail failed: {exc}", file=sys.stderr)
            print("The IAM policy needs bedrock:UpdateGuardrail "
                  "(infra/aws/medicure-setup-policy.json).", file=sys.stderr)
            return 1
        print(f"updated guardrail '{args.name}'")
        print(f"  id      : {existing['id']}")
        print(f"  grounding threshold {GROUNDING_THRESHOLD}, relevance {RELEVANCE_THRESHOLD}")
        print(f"\nBEDROCK_GUARDRAIL_ID={existing['id']}   (unchanged)")
        return 0

    try:
        response = client.create_guardrail(**config)
    except Exception as exc:  # noqa: BLE001
        print(f"error: create_guardrail failed: {exc}", file=sys.stderr)
        return 1

    guardrail_id = response["guardrailId"]
    print(f"created guardrail '{args.name}'")
    print(f"  id      : {guardrail_id}")
    print(f"  version : {response.get('version', 'DRAFT')}")
    print(f"  grounding threshold {GROUNDING_THRESHOLD}, relevance {RELEVANCE_THRESHOLD}")
    print(f"  denied topics: {', '.join(t['name'] for t in DENIED_TOPICS)}")
    print("\nAdd to .env:")
    print(f"BEDROCK_GUARDRAIL_ID={guardrail_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
