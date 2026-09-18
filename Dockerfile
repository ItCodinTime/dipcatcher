# syntax=docker/dockerfile:1

# --- Builder: resolve/sync the locked venv, then install the project ---
FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea AS builder
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app
RUN pip install --no-cache-dir uv==0.11.23
COPY pyproject.toml uv.lock README.md ./
# Dependencies first (cache-friendly), project last.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# --- Runtime: no uv, no pip, no build caches; self-contained venv ---
FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin dipcatcher
WORKDIR /app
COPY --from=builder --chown=dipcatcher:dipcatcher /app/.venv /app/.venv
COPY --chown=dipcatcher:dipcatcher src ./src
COPY --chown=dipcatcher:dipcatcher configs ./configs
USER dipcatcher
EXPOSE 8000
# Bind loopback by default. For 0.0.0.0, set QUANT_API_KEY and override CMD host.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f\"http://{os.environ.get('HOST', '127.0.0.1')}:{os.environ.get('PORT', '8000')}/health\", timeout=3)"
CMD ["dipcatcher", "api", "--host", "127.0.0.1", "--port", "8000"]
