"""FastAPI service — RAG, agent, LoRA classify, feedback logging."""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from tap_rag.agent.security_agent import SecurityAgent
from tap_rag.config import Settings, get_settings
from tap_rag.lora.classifier import classify_signal
from tap_rag.models.schemas import (
    AgentRequest,
    AgentResponse,
    AnomalyClassification,
    ClassifyRequest,
    FeedbackEvent,
    RAGQuery,
    RAGResponse,
)
from tap_rag.observability.metrics import AGENT_BLOCKS, RAG_LATENCY, RAG_REQUESTS
from tap_rag.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


@lru_cache
def get_pipeline() -> RAGPipeline:
    """Process-wide RAG singleton — Chroma + LLM are expensive to construct per request."""
    return RAGPipeline.from_settings(get_settings())


@lru_cache
def get_agent() -> SecurityAgent:
    """Process-wide LangGraph singleton — compile the graph once, reuse across requests."""
    return SecurityAgent()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm pipeline/agent caches; init logging and tracing."""
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    from tap_rag.observability.metrics import init_tracing

    init_tracing(settings.otel_exporter_otlp_endpoint)
    get_pipeline()
    get_agent()
    yield


app = FastAPI(
    title="TAP RAG Platform",
    version="1.0.0",
    description="Production RAG + LangGraph agent + LoRA classifier API",
    lifespan=lifespan,
)
_settings_boot = get_settings()
_cors_origins = (
    [f"http://localhost:{_settings_boot.streamlit_port}"]
    if _settings_boot.is_production
    else ["*"]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    response: Response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health")
def health(settings: Annotated[Settings, Depends(get_settings)]):
    """Liveness plus env/mock flags so operators can tell if Bedrock is actually in the path."""
    return {
        "status": "ok",
        "env": settings.app_env,
        "mock_llm": settings.use_mock_llm,
        "timestamp": datetime.now(UTC).isoformat(),
    }


@app.post("/v1/rag/query", response_model=RAGResponse)
def rag_query(
    request: RAGQuery,
    pipeline: Annotated[RAGPipeline, Depends(get_pipeline)],
):
    """Two-stage RAG: retrieve → re-rank → grounded generate."""
    try:
        with RAG_LATENCY.time():
            result = pipeline.query(request)
        RAG_REQUESTS.labels(status="ok").inc()
        return result
    except Exception as exc:  # noqa: BLE001
        RAG_REQUESTS.labels(status="error").inc()
        logger.exception("RAG query failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/agent/run", response_model=AgentResponse)
def agent_run(
    request: AgentRequest,
    agent: Annotated[SecurityAgent, Depends(get_agent)],
):
    """LangGraph security agent with read/write/execute guardrails."""
    result = agent.run(request)
    if result.blocked:
        AGENT_BLOCKS.inc()
    return result


@app.post("/v1/lora/classify", response_model=AnomalyClassification)
def lora_classify(payload: ClassifyRequest):
    """Anomaly classifier. Accepts `signal` (API) or `instruction` (training-notebook schema)."""
    return classify_signal(payload.signal)


@app.post("/v1/feedback")
def feedback(
    event: FeedbackEvent,
    settings: Annotated[Settings, Depends(get_settings)],
):
    """Persist feedback locally; mirror to S3 in production."""
    out_dir = Path("data/feedback")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%f')}.json"
    path.write_text(event.model_dump_json(indent=2))
    if settings.is_production:
        try:
            import boto3

            key = f"{settings.s3_feedback_prefix}{path.name}"
            boto3.client("s3", region_name=settings.aws_region).put_object(
                Bucket=settings.s3_bucket,
                Key=key,
                Body=path.read_bytes(),
                ContentType="application/json",
            )
        except Exception as exc:  # noqa: BLE001
            # Local file already stored — do not fail the request if the S3 mirror is down.
            logger.warning("S3 feedback upload failed: %s", exc)
    return {"stored": str(path)}


def run() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "tap_rag.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=not settings.is_production,
    )


if __name__ == "__main__":
    run()
