# Multi-stage build. The index artifacts are built at IMAGE BUILD TIME, not at
# container start: fitting TF-IDF over 253,973 rows takes ~15s, and doing it per
# container start makes every deploy and every autoscale event pay for it.
FROM python:3.12-slim AS builder

# Tesseract and the OpenCV runtime libs. `tesseract-ocr-hin` is included
# because Indian packaging is routinely bilingual, and running an English-only
# model over Devanagari yields confident garbage that then pollutes the token
# bag the resolver matches against.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .

COPY packages/ ./packages/
COPY apps/api/ ./apps/api/
COPY scripts/ ./scripts/
COPY data/processed/ ./data/processed/

RUN PYTHONPATH=/build python scripts/build_index.py --quiet

# The calibrator is COPIED, not refitted. Without it the service starts,
# reports "calibrator not fitted" on /v1/health, and every `probability` it
# returns is a raw similarity wearing a probability's name — exactly the
# failure calibrated abstention exists to prevent. The first Cloud Run deploy
# shipped that way, with a confident paracetamol match reporting
# calibrated=false.
#
# The obvious fix was `RUN fit_calibrator.py`, and it was wrong: fitting does
# thousands of searches over 253,973 rows, which takes minutes on a laptop and
# was still running after FIFTY on Cloud Build's single-vCPU worker. A trained
# model belongs in the image as an artifact, not as a build step.
#
# The trade is that the calibrator is now only as current as the checked-in
# file. It is fitted against index features, so changing normalize.py or the
# catalogues invalidates BOTH (see NOTES.md) — rebuild the index and re-run
# scripts/fit_calibrator.py locally, then rebuild the image.
COPY data/artifacts/calibrator.joblib data/artifacts/calibration_report.json ./data/artifacts/

# --- runtime ---
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin \
        libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root. The service reads datasets and writes nothing.
RUN useradd --create-home --uid 10001 medicure
WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder --chown=medicure:medicure /build/packages ./packages
COPY --from=builder --chown=medicure:medicure /build/apps/api ./apps/api
COPY --from=builder --chown=medicure:medicure /build/scripts ./scripts
COPY --from=builder --chown=medicure:medicure /build/data ./data

USER medicure

# PORT is what the runtime asks us to listen on. Cloud Run injects it (8080)
# and routes nothing to a container listening elsewhere — the revision simply
# fails its health check with no useful message. The default keeps `docker run`
# and docker-compose on 8000 as before.
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1 PORT=8000

EXPOSE 8000

# The index takes a couple of seconds to load, so the check starts late enough
# not to kill a healthy container during startup. Cloud Run ignores Docker's
# HEALTHCHECK and uses its own probes; this is for compose and plain docker.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/v1/health" || exit 1

# Shell form so ${PORT} expands, and `exec` so uvicorn becomes PID 1 and
# receives SIGTERM directly. Without exec, the shell holds PID 1, swallows the
# signal, and every deploy waits out the full termination grace period before
# the old revision dies.
#
# One worker. Each worker loads its own copy of the 125 MB index, so on a 2 GB
# instance a second worker costs more than it returns.
CMD exec python -m uvicorn apps.api.main:app --host 0.0.0.0 --port "${PORT}" --workers 1
