"""Shared Pydantic domain models — validation at every boundary."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ActionType(str, Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"


class ValidationResult(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"


class AnomalyLabel(str, Enum):
    SUSPICIOUS = "SUSPICIOUS"
    BENIGN = "BENIGN"
    INVESTIGATE = "INVESTIGATE"


class Verdict(str, Enum):
    MALICIOUS = "malicious"
    SUSPICIOUS = "suspicious"
    CLEAN = "clean"


# ── RAG ──


class DocumentChunk(BaseModel):
    content: str
    source: str
    chunk_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float | None = None


class RAGQuery(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    session_id: str | None = None


class Citation(BaseModel):
    source: str
    excerpt: str = Field(max_length=500)
    score: float | None = None


class RAGResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    model_id: str
    latency_ms: float | None = None
    session_id: str | None = None


class FeedbackEvent(BaseModel):
    question: str
    answer: str
    rating: Literal["up", "down"] | None = None
    score: int | None = Field(default=None, ge=1, le=5)
    session_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── MCP / Reputation ──


class HashReputationRequest(BaseModel):
    hash_value: str = Field(min_length=8, max_length=128)
    first_seen_after: str | None = None
    last_seen_before: str | None = None

    @field_validator("hash_value")
    @classmethod
    def must_be_hex(cls, v: str) -> str:
        cleaned = v.lower().strip()
        if not all(c in "0123456789abcdef" for c in cleaned):
            raise ValueError("hash_value must be hexadecimal")
        return cleaned


class HashReputationResult(BaseModel):
    hash: str
    verdict: Verdict
    threat_score: int = Field(ge=0, le=100)
    first_seen: str
    last_seen: str
    prevalence: int
    family: str | None = None
    error: str | None = None


class IPReputationRequest(BaseModel):
    ip_address: str = Field(min_length=3, max_length=64)


class IPReputationResult(BaseModel):
    ip: str
    verdict: Verdict
    threat_score: int = Field(ge=0, le=100)
    country: str
    asn: str
    org: str
    associated_campaigns: list[str] = Field(default_factory=list)
    error: str | None = None


class BulkHashRequest(BaseModel):
    hash_values: list[str] = Field(min_length=1, max_length=100)


# ── LoRA / Anomaly classification ──


class AnomalyClassification(BaseModel):
    classification: AnomalyLabel
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    routed_to_analyst: bool = False

    @model_validator(mode="after")
    def apply_routing(self) -> AnomalyClassification:
        # Threshold applied by caller; keep field for API responses.
        return self


class TrainingExample(BaseModel):
    instruction: str
    output: str

    @field_validator("output")
    @classmethod
    def must_contain_classification(cls, v: str) -> str:
        if "Classification:" not in v:
            raise ValueError("output must contain Classification: line")
        return v


# ── Agent ──


class AgentRequest(BaseModel):
    query: str = Field(min_length=3, max_length=4000)
    session_id: str | None = None


class AgentResponse(BaseModel):
    message: str
    action_type: ActionType | None = None
    validation_result: ValidationResult | None = None
    rejection_reason: str | None = None
    proposed_diff: str | None = None
    blocked: bool = False


# ── Eval ──


class GoldenExample(BaseModel):
    id: str
    question: str
    expected_answer_contains: list[str] = Field(default_factory=list)
    expected_sources: list[str] = Field(default_factory=list)
    must_not_contain: list[str] = Field(default_factory=list)
    category: str = "general"
    difficulty: Literal["easy", "medium", "hard"] = "medium"


class EvalMetrics(BaseModel):
    n_examples: int
    answer_hit_rate: float
    citation_precision: float
    citation_recall: float
    hallucination_rate: float
    latency_p50_ms: float
    latency_p95_ms: float
    pass_threshold: bool
    per_example: list[dict[str, Any]] = Field(default_factory=list)
