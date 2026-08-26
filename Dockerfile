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
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1

EXPOSE 8000

# The index takes a couple of seconds to load, so the check starts late enough
# not to kill a healthy container during startup.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/v1/health || exit 1

# One worker by default. Each worker loads its own copy of the 125 MB index, so
# on a 2 GB instance a second worker costs more than it returns.
CMD ["python", "-m", "uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
