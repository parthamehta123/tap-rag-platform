"""LangChain LCEL RAG pipeline: retrieve → re-rank → grounded generate."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from tap_rag.config import Settings, get_settings
from tap_rag.models.schemas import Citation, RAGQuery, RAGResponse
from tap_rag.rag.ingest import ingest, load_vectorstore
from tap_rag.rag.llm import LLMClient, format_context, get_llm, grounded_system_prompt
from tap_rag.rag.retriever import TwoStageRetriever

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Composable RAG service — inject mocks for unit tests (DIP)."""

    def __init__(
        self,
        retriever: TwoStageRetriever,
        llm: LLMClient,
        settings: Settings | None = None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.settings = settings or get_settings()

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        rebuild_index: bool = False,
    ) -> RAGPipeline:
        settings = settings or get_settings()
        if rebuild_index or not Path(settings.chroma_persist_dir).exists():
            vs = ingest(settings=settings)
        else:
            try:
                vs = load_vectorstore(settings)
                if vs._collection.count() == 0:  # noqa: SLF001
                    vs = ingest(settings=settings)
            except Exception:  # noqa: BLE001
                vs = ingest(settings=settings)
        retriever = TwoStageRetriever(vs, settings, enable_rerank=False)
        # Re-rank optional: enable when sentence-transformers installed & not in CI mock mode
        if not settings.use_mock_embeddings:
            retriever = TwoStageRetriever(vs, settings, enable_rerank=True)
        return cls(retriever=retriever, llm=get_llm(settings), settings=settings)

    def _build_messages(self, question: str, docs: list[Document]) -> list:
        context = format_context(docs)
        user = (
            f"<context>\n{context}\n</context>\n\n"
            f"Question: {question}\n\n"
            "Answer based solely on the context. Cite sources."
        )
        return [grounded_system_prompt(), HumanMessage(content=user)]

    def _extract_citations(self, answer: str, docs: list[Document]) -> list[Citation]:
        citations: list[Citation] = []
        seen: set[str] = set()
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            if source in seen:
                continue
            seen.add(source)
            citations.append(
                Citation(
                    source=source,
                    excerpt=doc.page_content[:300],
                    score=doc.metadata.get("rerank_score")
                    or doc.metadata.get("vector_score"),
                )
            )
        # Prefer Sources: line if present
        sources_line = re.search(r"Sources?:\s*(.+)", answer, re.IGNORECASE)
        if sources_line and not citations:
            for part in re.split(r"[,;]", sources_line.group(1)):
                part = part.strip()
                if part:
                    citations.append(Citation(source=part, excerpt=""))
        return citations

    def query(self, request: RAGQuery) -> RAGResponse:
        start = time.perf_counter()
        docs = self.retriever.retrieve(request.question)
        messages = self._build_messages(request.question, docs)
        ai = self.llm.invoke(messages)
        answer = str(ai.content)
        latency_ms = (time.perf_counter() - start) * 1000
        return RAGResponse(
            answer=answer,
            citations=self._extract_citations(answer, docs),
            model_id=(
                "mock-llm"
                if self.settings.use_mock_llm
                else self.settings.bedrock_llm_model_id
            ),
            latency_ms=latency_ms,
            session_id=request.session_id,
        )

    def stream(self, request: RAGQuery):
        docs = self.retriever.retrieve(request.question)
        messages = self._build_messages(request.question, docs)
        yield from self.llm.stream_text(messages)
