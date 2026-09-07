"""RAG package exports."""

from tap_rag.rag.ingest import ingest, load_vectorstore
from tap_rag.rag.pipeline import RAGPipeline
from tap_rag.rag.retriever import TwoStageRetriever, deduplicate_by_source

__all__ = [
    "RAGPipeline",
    "TwoStageRetriever",
    "deduplicate_by_source",
    "ingest",
    "load_vectorstore",
]
