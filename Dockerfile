# syntax=docker/dockerfile:1.7

FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --create-home appuser

COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
COPY configs ./configs

RUN pip install --upgrade pip && pip install "."

RUN chown -R appuser:appuser /app
USER appuser

# ── API image ──
FROM base AS api
ENV USE_MOCK_LLM=false \
    USE_MOCK_EMBEDDINGS=false \
    APP_ENV=production
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "tap_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ── Streamlit UI image ──
FROM base AS ui
ENV USE_MOCK_LLM=false \
    USE_MOCK_EMBEDDINGS=false \
    APP_ENV=production
EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "src/tap_rag/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# ── Ingest / eval worker ──
FROM base AS worker
CMD ["python", "-m", "tap_rag.rag.ingest"]

# ── MCP server ──
FROM base AS mcp
CMD ["python", "-m", "tap_rag.mcp.server"]
