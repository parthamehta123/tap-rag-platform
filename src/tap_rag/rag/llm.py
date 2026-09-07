"""LLM and embedding abstractions — DIP for testability without live AWS."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from tap_rag.config import Settings, get_settings


class LLMClient(ABC):
    @abstractmethod
    def invoke(self, messages: list[BaseMessage]) -> AIMessage: ...

    @abstractmethod
    def stream_text(self, messages: list[BaseMessage]) -> Iterable[str]: ...


class MockLLMClient(LLMClient):
    """Deterministic LLM for local/CI — no Bedrock calls."""

    def __init__(self, default_reply: str | None = None):
        self.default_reply = default_reply or (
            "Based on the provided context, the TapCloudOps ClickHouse "
            "dev cluster is reached via `clickhouse-client --host "
            "ch-dev.tap.internal --port 9000` with credentials in Vault "
            "path `secret/tap/clickhouse/dev`. Sources: docs/clickhouse.md"
        )

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        user = next(
            (m.content for m in reversed(messages) if isinstance(m, HumanMessage)),
            "",
        )
        text = str(user).lower()
        system_blob = " ".join(
            str(m.content).lower() for m in messages if isinstance(m, SystemMessage)
        )
        is_agent = "classify your intended action" in system_blob

        # RAG path: echo grounded context so golden-eval keywords match
        if not is_agent and "<context>" in text and "</context>" in text:
            ctx = text.split("<context>", 1)[1].split("</context>", 1)[0]
            question = ""
            if "question:" in text:
                question = text.split("question:", 1)[1].strip()
            return AIMessage(
                content=(
                    f"Based on the provided context: {ctx[:1200].strip()}\n\n"
                    f"Sources: see citations.\n"
                    f"(Answered: {question[:200]})"
                )
            )

        if any(w in text for w in ("delete", "modify", "change", "update", "write", "remove")):
            return AIMessage(content="write\nPlan: prepare a sandboxed change")
        if any(w in text for w in ("run", "execute", "restart", "deploy")):
            return AIMessage(content="execute\nPlan: run a sandboxed command")
        if "terraform" in text or "geoip" in text:
            return AIMessage(
                content=(
                    "read\nThe GeoIP Terraform module lives under "
                    "modules/geoip and uses aws_route53 + S3 for MMDB sync. "
                    "Sources: docs/terraform-geoip.md"
                )
            )
        return AIMessage(content=f"read\n{self.default_reply}")

    def stream_text(self, messages: list[BaseMessage]) -> Iterable[str]:
        msg = self.invoke(messages)
        for word in str(msg.content).split(" "):
            yield word + " "


class BedrockLLMClient(LLMClient):
    def __init__(self, settings: Settings | None = None):
        from langchain_aws import ChatBedrock

        self.settings = settings or get_settings()
        kwargs: dict = {
            "model_id": self.settings.bedrock_llm_model_id,
            "region_name": self.settings.aws_region,
            "model_kwargs": {"temperature": 0.3},
        }
        if self.settings.bedrock_guardrail_id:
            kwargs["guardrails"] = {
                "guardrailIdentifier": self.settings.bedrock_guardrail_id,
                "guardrailVersion": self.settings.bedrock_guardrail_version,
            }
        self._llm: BaseChatModel = ChatBedrock(**kwargs)

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        result = self._llm.invoke(messages)
        return AIMessage(content=result.content)

    def stream_text(self, messages: list[BaseMessage]) -> Iterable[str]:
        for chunk in self._llm.stream(messages):
            if chunk.content:
                yield str(chunk.content)


class MockEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings for local/CI (no Bedrock)."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = text.lower().replace("/", " ").replace(".", " ").split()
        for tok in tokens:
            h = hash(tok) % self.dim
            vec[h] += 1.0
            # bigrams
            if len(tok) > 3:
                for i in range(len(tok) - 2):
                    bg = tok[i : i + 3]
                    vec[hash(bg) % self.dim] += 0.5
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def get_llm(settings: Settings | None = None) -> LLMClient:
    settings = settings or get_settings()
    if settings.use_mock_llm:
        return MockLLMClient()
    return BedrockLLMClient(settings)


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    settings = settings or get_settings()
    if settings.use_mock_embeddings:
        return MockEmbeddings()
    from langchain_aws import BedrockEmbeddings

    return BedrockEmbeddings(
        model_id=settings.bedrock_embed_model_id,
        region_name=settings.aws_region,
    )


def grounded_system_prompt() -> SystemMessage:
    return SystemMessage(
        content=(
            "You are a TAP CloudOps assistant. Base your entire response solely on "
            "the information provided in the context. If the context is insufficient, "
            "say you do not know. End with a Sources: line listing source file paths."
        )
    )


def format_context(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] source={source}\n{doc.page_content}")
    return "\n\n".join(parts)
