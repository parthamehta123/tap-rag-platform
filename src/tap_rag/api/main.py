"""FastAPI service — RAG, agent, LoRA classify, feedback logging."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from tap_rag.agent.security_agent import SecurityAgent
from tap_rag.config import Settings, get_settings
from tap_rag.lora.classifier import classify_signal
from tap_rag.models.schemas import (
    AgentRequest,
    AgentResponse,
    AnomalyClassification,
    FeedbackEvent,
    RAGQuery,
    RAGResponse,
)
from tap_rag.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


@lru_cache
def get_pipeline() -> RAGPipeline:
    return RAGPipeline.from_settings(get_settings())


@lru_cache
def get_agent() -> SecurityAgent:
    return SecurityAgent()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=get_settings().log_level)
    # Warm caches
    get_pipeline()
    get_agent()
    yield


app = FastAPI(
    title="TAP RAG Platform",
    version="1.0.0",
    description="Production RAG + LangGraph agent + LoRA classifier API",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health(settings: Annotated[Settings, Depends(get_settings)]):
    return {
        "status": "ok",
        "env": settings.app_env,
        "mock_llm": settings.use_mock_llm,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/v1/rag/query", response_model=RAGResponse)
def rag_query(
    request: RAGQuery,
    pipeline: Annotated[RAGPipeline, Depends(get_pipeline)],
):
    try:
        return pipeline.query(request)
    except Exception as exc:  # noqa: BLE001
        logger.exception("RAG query failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/agent/run", response_model=AgentResponse)
def agent_run(
    request: AgentRequest,
    agent: Annotated[SecurityAgent, Depends(get_agent)],
):
    return agent.run(request)


@app.post("/v1/lora/classify", response_model=AnomalyClassification)
def lora_classify(payload: dict):
    signal = payload.get("signal") or payload.get("instruction")
    if not signal:
        raise HTTPException(status_code=422, detail="signal required")
    return classify_signal(signal)


@app.post("/v1/feedback")
def feedback(
    event: FeedbackEvent,
    settings: Annotated[Settings, Depends(get_settings)],
):
    """Log {question, answer, rating} for RLHF / eval feedback loops."""
    out_dir = Path("data/feedback")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}.json"
    path.write_text(event.model_dump_json(indent=2))
    # Production: also write to S3 via boto3 when not local
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
