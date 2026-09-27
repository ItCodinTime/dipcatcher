# syntax=docker/dockerfile:1

# Read-only research evidence API (quant_fund.api.research_api).
# Serves committed receipts/, verifier/, artifacts/ plus the gitignored
# data/metadata tree — mount a data root at runtime:
#
#   docker build -f docker/research-api.Dockerfile -t dipcatcher-research-api .
#   docker run --rm -p 127.0.0.1:8010:8010 \
#     -v /path/to/data:/app/data:ro dipcatcher-research-api
#
# Loopback-only by default. To expose beyond the host, set RESEARCH_API_KEY
# and override the CMD host — never without the key. The service has no
# trading, order, broker, or execution endpoints.

# --- Builder: resolve/sync the locked venv, then install the project ---
FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea AS builder
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app
RUN pip install --no-cache-dir uv==0.11.23
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# --- Runtime: no uv, no pip, no build caches; self-contained venv ---
FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    RESEARCH_API_DATA_ROOT=/app/data \
    RESEARCH_API_RECEIPTS_DIR=/app/receipts \
    RESEARCH_API_VERIFIER_DIR=/app/verifier \
    RESEARCH_API_ARTIFACTS_DIR=/app/artifacts
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin dipcatcher
WORKDIR /app
COPY --from=builder --chown=dipcatcher:dipcatcher /app/.venv /app/.venv
COPY --chown=dipcatcher:dipcatcher src ./src
COPY --chown=dipcatcher:dipcatcher receipts ./receipts
COPY --chown=dipcatcher:dipcatcher verifier ./verifier
COPY --chown=dipcatcher:dipcatcher artifacts ./artifacts
# data/metadata is dockerignored — mount it read-only at runtime (see header).
RUN mkdir -p /app/data && chown dipcatcher:dipcatcher /app/data
USER dipcatcher
EXPOSE 8010
# Bind loopback by default. For 0.0.0.0, set RESEARCH_API_KEY and override CMD host.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f\"http://{os.environ.get('HOST', '127.0.0.1')}:{os.environ.get('PORT', '8010')}/health\", timeout=3)"
CMD ["python", "-m", "uvicorn", "quant_fund.api.research_api:app", "--host", "127.0.0.1", "--port", "8010"]
