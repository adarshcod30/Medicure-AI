#!/usr/bin/env bash
# Find out which Bedrock models this account can ACTUALLY invoke.
#
#   bash infra/aws/probe_models.sh
#
# Listing a model proves nothing — an account with no payment instrument lists
# everything and invokes none of it. This calls each candidate for real and
# reports the outcome, so the answer is empirical rather than assumed.
#
# Ordered first-party first. Amazon Nova and Titan are native AWS services;
# the third-party families are delivered via AWS Marketplace subscriptions,
# which is the thing that requires a valid payment method. If billing is
# unresolved, the first-party models are the ones likely to work.
#
# The third-party model IDs below are deliberately kept: this script exists to
# test what an account can actually invoke, and dropping them would remove the
# evidence for choosing Nova.

set -uo pipefail
REGION="${AWS_REGION:-us-east-1}"

CANDIDATES=(
  "amazon.nova-pro-v1:0"
  "amazon.nova-lite-v1:0"
  "amazon.nova-micro-v1:0"
  "us.amazon.nova-pro-v1:0"
  "us.amazon.nova-lite-v1:0"
  "us.amazon.nova-micro-v1:0"
  "amazon.titan-text-premier-v1:0"
  "amazon.titan-text-express-v1"
  "us.anthropic.claude-sonnet-4-6"
  "us.anthropic.claude-haiku-4-5-20251001-v1:0"
  "us.meta.llama3-3-70b-instruct-v1:0"
  "mistral.mistral-large-2407-v1:0"
)

printf "Probing Bedrock in %s — invoking each model for real\n\n" "$REGION"
printf "  %-46s %s\n" "MODEL" "RESULT"
printf "  %-46s %s\n" "----------------------------------------------" "------"

WORKING=()
for model in "${CANDIDATES[@]}"; do
  out=$(aws bedrock-runtime converse --region "$REGION" --model-id "$model" \
        --messages '[{"role":"user","content":[{"text":"Reply with the single word OK"}]}]' \
        --inference-config '{"maxTokens":16}' --output text 2>&1)

  if echo "$out" | grep -qi "ok"; then
    printf "  %-46s \033[32mWORKS\033[0m\n" "$model"
    WORKING+=("$model")
  elif echo "$out" | grep -q "INVALID_PAYMENT_INSTRUMENT"; then
    printf "  %-46s \033[31mbilling\033[0m  (Marketplace subscription needs a payment method)\n" "$model"
  elif echo "$out" | grep -qi "don't have access\|AccessDenied"; then
    printf "  %-46s \033[33mno access\033[0m  (enable in Bedrock console)\n" "$model"
  elif echo "$out" | grep -qi "not found\|ValidationException"; then
    printf "  %-46s \033[90mnot available here\033[0m\n" "$model"
  else
    printf "  %-46s \033[31mfailed\033[0m  %s\n" "$model" "$(echo "$out" | head -1 | cut -c1-60)"
  fi
done

echo
if [ ${#WORKING[@]} -eq 0 ]; then
  echo "Nothing is invocable. If every line says 'billing', add a payment method:"
  echo "  https://console.aws.amazon.com/billing/home#/paymentpreferences"
  echo "If they say 'no access', enable the models:"
  echo "  https://console.aws.amazon.com/bedrock/home?region=$REGION#/modelaccess"
  exit 1
fi

echo "Working models (${#WORKING[@]}):"
for m in "${WORKING[@]}"; do echo "  $m"; done
echo
echo "Put the best one in .env as BEDROCK_MODEL_ID, and the cheapest as"
echo "BEDROCK_FAST_MODEL_ID. The pipeline is model-agnostic — see below."
