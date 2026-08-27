#!/usr/bin/env bash
# Verify the AWS side of MediCure end to end.
#   bash infra/aws/verify.sh
#
# Checks in dependency order and stops at the first real blocker, so the output
# names one thing to fix rather than a wall of red.

set -uo pipefail
REGION="${AWS_REGION:-us-east-1}"
SONNET="${BEDROCK_MODEL_ID:-us.amazon.nova-pro-v1:0}"
HAIKU="${BEDROCK_FAST_MODEL_ID:-us.amazon.nova-lite-v1:0}"
TITAN="amazon.titan-embed-text-v2:0"

ok()   { printf "  \033[32mOK\033[0m    %s\n" "$1"; }
bad()  { printf "  \033[31mFAIL\033[0m  %s\n" "$1"; }
info() { printf "        %s\n" "$1"; }

echo "MediCure AWS check  (region: $REGION)"
echo

# 1 — credentials
who=$(aws sts get-caller-identity --output json 2>&1)
if echo "$who" | grep -q '"Account"'; then
  acct=$(echo "$who" | python3 -c 'import json,sys;print(json.load(sys.stdin)["Account"])')
  arn=$(echo "$who"  | python3 -c 'import json,sys;print(json.load(sys.stdin)["Arn"])')
  ok "credentials valid  (account $acct)"
  case "$arn" in
    *":root") info "WARNING: these are ROOT credentials. Create an IAM user instead." ;;
    *)        info "identity: $arn" ;;
  esac
else
  bad "no valid credentials"
  info "$(echo "$who" | head -1)"
  info "Fix: aws configure   (or: aws login)"
  exit 1
fi

# 2 — Bedrock control plane
if aws bedrock list-foundation-models --region "$REGION" >/dev/null 2>&1; then
  n=$(aws bedrock list-foundation-models --region "$REGION" --by-provider amazon \
      --query 'length(modelSummaries)' --output text 2>/dev/null)
  ok "Bedrock reachable   ($n Amazon models listed)"
else
  bad "cannot list Bedrock models"
  info "Fix: attach infra/aws/medicure-bedrock-policy.json to this identity."
  exit 1
fi

# 3 — actual invocation. This is where billing and model access show up.
for pair in "primary:$SONNET" "fast:$HAIKU"; do
  label="${pair%%:*}"; model="${pair#*:}"
  out=$(aws bedrock-runtime converse --region "$REGION" --model-id "$model" \
        --messages '[{"role":"user","content":[{"text":"Reply with OK"}]}]' \
        --inference-config '{"maxTokens":16}' --output text 2>&1)
  if echo "$out" | grep -qi "ok"; then
    ok "$label invocable      ($model)"
  elif echo "$out" | grep -q "INVALID_PAYMENT_INSTRUMENT"; then
    bad "$label BLOCKED       AWS Marketplace subscription refused"
    info "Seen on AISPL (AWS India, INR billing) accounts even with a valid card."
    info "Nova is first-party AWS and should NOT hit this — if it does, the"
    info "account has a broader Marketplace restriction. Check:"
    info "  console.aws.amazon.com/billing/home#/paymentpreferences"
    exit 1
  elif echo "$out" | grep -q "AccessDenied"; then
    bad "$label access denied"
    info "Fix: request model access at console.aws.amazon.com/bedrock -> Model access"
    exit 1
  else
    bad "$label failed"
    info "$(echo "$out" | head -1 | cut -c1-140)"
  fi
done

# 4 — embeddings, needed for the M3 retrieval work
if aws bedrock-runtime invoke-model --region "$REGION" --model-id "$TITAN" \
     --body '{"inputText":"test"}' --cli-binary-format raw-in-base64-out \
     /dev/null >/dev/null 2>&1; then
  ok "Titan embeddings invocable"
else
  info "Titan embeddings not invocable (only needed for M3 vector search)"
fi

echo
echo "All green. Run the API with ENABLE_BEDROCK=true."
