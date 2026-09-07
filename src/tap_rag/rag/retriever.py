"""Two-stage retrieval: vector search → source dedupe → cross-encoder re-rank."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Protocol

from langchain_core.documents import Document

from tap_rag.config import Settings, get_settings

logger = logging.getLogger(__name__)


class VectorStoreLike(Protocol):
    def similarity_search_with_score(
        self, query: str, k: int = 5
    ) -> list[tuple[Document, float]]: ...


def deduplicate_by_source(
    docs: list[Document],
    max_per_source: int = 2,
) -> list[Document]:
    """Cap chunks per source file so one verbose doc cannot monopolize top-k."""
    counts: dict[str, int] = defaultdict(int)
    kept: list[Document] = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        if counts[source] >= max_per_source:
            continue
        counts[source] += 1
        kept.append(doc)
    return kept


class CrossEncoderReranker:
    """Optional cross-encoder re-ranker. Falls back to original order if unavailable."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cross-encoder unavailable (%s); skipping re-rank", exc)
            self._model = False

    def rerank(self, query: str, docs: list[Document], top_n: int) -> list[Document]:
        if not docs:
            return []
        self._load()
        if self._model is False or self._model is None:
            return docs[:top_n]

        pairs = [(query, d.page_content) for d in docs]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
        result = []
        for doc, score in ranked[:top_n]:
            doc.metadata["rerank_score"] = float(score)
            result.append(doc)
        return result


class TwoStageRetriever:
    def __init__(
        self,
        vectorstore: VectorStoreLike,
        settings: Settings | None = None,
        enable_rerank: bool = True,
    ):
        self.vectorstore = vectorstore
        self.settings = settings or get_settings()
        self.reranker = (
            CrossEncoderReranker(self.settings.cross_encoder_model)
            if enable_rerank
            else None
        )

    def retrieve(self, query: str) -> list[Document]:
        # Stage 1 — bi-encoder vector search
        pairs = self.vectorstore.similarity_search_with_score(
            query, k=self.settings.retriever_top_k
        )
        docs = []
        for doc, score in pairs:
            # Chroma cosine distance: lower is better; store inverted similarity-ish
            doc.metadata["vector_score"] = float(score)
            docs.append(doc)

        docs = deduplicate_by_source(docs, self.settings.max_chunks_per_source)

        # Stage 2 — cross-encoder re-ranking
        if self.reranker:
            docs = self.reranker.rerank(query, docs, self.settings.rerank_top_n)
        else:
            docs = docs[: self.settings.rerank_top_n]
        return docs
