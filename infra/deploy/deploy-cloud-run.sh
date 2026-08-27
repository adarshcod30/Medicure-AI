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
MAX_INSTANCES=3

# --- the two settings that decide whether it crashes ---------------------
#
# Measured on this machine: 806 MB resident once the index is loaded, peaking
# at 1,536 MB during an image scan (DIP holds several full-resolution
# intermediates). 1 GiB OOMs on the main use case; 2 GiB leaves ~512 MB.
MEMORY=2Gi
CPU=1

# Cloud Run defaults to 80 concurrent requests per instance. Each in-flight
# scan costs ~730 MB on top of the shared 806 MB index, so two at once need
# ~2.3 GB and the instance is killed. One request per instance is the honest
# setting for this workload; concurrency is bought with memory, not wishes.
CONCURRENCY=1

# A scan is ~5s warm. 120s is generous enough for a cold start plus a large
# upload, and bounded so a hung request cannot bill indefinitely.
TIMEOUT=120

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
