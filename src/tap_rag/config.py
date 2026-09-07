"""Central configuration via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "local"
    log_level: str = "INFO"
    use_mock_llm: bool = True
    use_mock_embeddings: bool = True

    aws_region: str = "us-east-1"
    bedrock_llm_model_id: str = "us.anthropic.claude-sonnet-4-6"
    bedrock_embed_model_id: str = "amazon.titan-embed-text-v2:0"
    bedrock_guardrail_id: str = ""
    bedrock_guardrail_version: str = "DRAFT"

    docs_dir: Path = Path("./data/docs")
    chroma_persist_dir: Path = Path("./chroma-db")
    chroma_collection: str = "tap_docs"
    chunk_size: int = 600
    chunk_overlap: int = 100
    retriever_top_k: int = 5
    rerank_top_n: int = 3
    max_chunks_per_source: int = 2
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    streamlit_port: int = 8501

    s3_bucket: str = "tap-rag-platform-dev"
    s3_chroma_prefix: str = "chroma/"
    s3_feedback_prefix: str = "feedback/"
    s3_eval_prefix: str = "evals/"

    lora_adapter_dir: Path = Path("./lora-adapters")
    lora_base_model: str = "meta-llama/Llama-2-7b-chat-hf"
    lora_confidence_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    hf_token: str = ""

    agent_max_iterations: int = 15
    agent_diff_delete_threshold: int = 50

    otel_exporter_otlp_endpoint: str = ""
    prometheus_port: int = 9090

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() in {"prod", "production"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
