#!/usr/bin/env bash
#
# Deploy the MediCure API to Cloud Run, with the cost and memory settings that
# were measured rather than guessed. Run from the repository root.
#
#     bash infra/deploy/deploy-cloud-run.sh YOUR_PROJECT_ID
#
# Every flag below that carries a cost or a crash risk is fixed in this script
# on purpose. See infra/deploy/cloud-run.md for the measurements behind them.

set -euo pipefail

PROJECT="${1:-}"
REGION="${REGION:-asia-south1}"
REPO="${REPO:-medicure}"
SERVICE="${SERVICE:-medicure-api}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/api:latest"

if [[ -z "$PROJECT" ]]; then
  echo "usage: bash infra/deploy/deploy-cloud-run.sh YOUR_PROJECT_ID" >&2
  echo "  find yours with: gcloud projects list" >&2
  exit 1
fi

# --- the two settings that decide the bill -------------------------------
#
# MIN_INSTANCES=0 is the difference between ~Rs 11/month and ~$70/month.
# At 0 the service scales to zero and you are billed only while a request is
# actually running; the free tier (180,000 vCPU-s + 360,000 GiB-s per month)
# then covers roughly 36,000 scans. At 1 the container bills 24/7 whether
# anyone visits or not: 2 GiB + 1 vCPU continuously is about $70 a month.
#
# The trade is cold starts. Do not "fix" a slow first request by raising this.
MIN_INSTANCES=0

# Bounds the blast radius. Without a cap, a crawler or a retry loop can fan out
# to Cloud Run's default 100 instances and spend real money before you notice.
MAX_INSTANCES=5

# --- the two settings that decide whether it crashes ---------------------
#
# Measured: 806 MB resident once the index is loaded, peaking at 1,244 MB
# during an image scan (DIP holds several full-resolution intermediates; the
# peak was 1,536 MB before the OCR fan-out was parallelised). 1 GiB OOMs on the
# main use case.
#
# CPU=4 is not optional. At 1 vCPU a scan took 100-120s against 11s locally and
# blew through the request timeout — every real photo returned HTTP 504. The
# pipeline is OCR- and OpenCV-heavy and Cloud Run's shared vCPU is roughly 10x
# slower than a laptop core.
MEMORY=2Gi
CPU=4

# Cloud Run defaults to 80 concurrent requests per instance, which would OOM
# instantly. But the reason this stays at 1 is latency, not memory: after
# parallelising OCR and the orientation probes, a scan uses all 4 vCPUs. Two
# concurrent scans on one instance would split them and each take roughly twice
# as long for the same total throughput.
#
# Memory would now allow it — a scan costs ~438 MB above the shared 806 MB
# index, so two fit in 2 GiB. Capacity comes from MAX_INSTANCES instead: each
# concurrent user gets an instance with all four cores, and Cloud Run only
# bills instances that exist. Waiting 25s is acceptable; waiting 50s because
# someone else clicked at the same moment is not.
CONCURRENCY=1

# A warm scan is ~25s on this hardware. 300s covers a cold start (image pull
# plus a 2s index load) and a large upload, and is bounded so a hung request
# cannot bill indefinitely. The earlier 120s was not enough and returned 504.
TIMEOUT=300

echo "project      : $PROJECT"
echo "region       : $REGION"
echo "image        : $IMAGE"
echo "memory/cpu   : $MEMORY / $CPU vCPU, concurrency $CONCURRENCY"
echo "instances    : min=$MIN_INSTANCES max=$MAX_INSTANCES  <- min=0 keeps this free"
echo

gcloud config set project "$PROJECT" >/dev/null

echo "==> enabling required APIs (idempotent)"
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com

echo "==> ensuring Artifact Registry repository '$REPO' exists"
gcloud artifacts repositories describe "$REPO" --location="$REGION" >/dev/null 2>&1 || \
  gcloud artifacts repositories create "$REPO" \
    --repository-format=docker --location="$REGION" \
    --description="MediCure API images"

echo "==> building image with Cloud Build"
# .gcloudignore keeps this upload near 40 MB by excluding apps/web and data/raw.
gcloud builds submit --tag "$IMAGE" .

echo "==> deploying to Cloud Run"
gcloud run deploy "$SERVICE" \
  --image "$IMAGE" \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --memory "$MEMORY" \
  --cpu "$CPU" \
  --concurrency "$CONCURRENCY" \
  --timeout "$TIMEOUT" \
  --min-instances "$MIN_INSTANCES" \
  --max-instances "$MAX_INSTANCES" \
  --cpu-boost \
  --set-env-vars "ENVIRONMENT=production,ENABLE_BEDROCK=true,STORE_UPLOADS=false" \
  --set-secrets "JWT_SECRET=medicure-jwt-secret:latest,MONGODB_URI=medicure-mongodb-uri:latest,AWS_ACCESS_KEY_ID=medicure-aws-key-id:latest,AWS_SECRET_ACCESS_KEY=medicure-aws-secret:latest,BEDROCK_GUARDRAIL_ID=medicure-guardrail-id:latest"

URL=$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)')
echo
echo "deployed: $URL"
echo
echo "verify (first request is a cold start, 15-40s):"
echo "  curl -s $URL/v1/health | python3 -m json.tool"
echo
echo "confirm it will actually scale to zero:"
echo "  gcloud run services describe $SERVICE --region $REGION \\"
echo "    --format='value(spec.template.metadata.annotations)' | tr ',' '\\n' | grep -i instances"
