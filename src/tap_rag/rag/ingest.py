"""Document ingestion: discover → load → chunk → embed → ChromaDB."""

from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from tap_rag.config import Settings, get_settings
from tap_rag.rag.llm import get_embeddings

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".md", ".py", ".go", ".scala", ".sh", ".tf", ".yaml", ".yml", ".pdf", ".txt"}

CODE_SEPARATORS = ["\nclass ", "\ndef ", "\nfunc ", "\n\n", "\n", " ", ""]
PROSE_SEPARATORS = ["\n\n", "\n", " ", ""]


def discover_files(docs_dir: Path) -> list[Path]:
    files: list[Path] = []
    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")
    for path in docs_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return sorted(files)


def load_document(path: Path) -> list[Document]:
    if path.suffix.lower() == ".pdf":
        docs = PyPDFLoader(str(path)).load()
    else:
        docs = TextLoader(str(path), encoding="utf-8").load()
    for doc in docs:
        doc.metadata["source"] = str(path)
        doc.metadata["content_hash"] = hashlib.sha256(
            doc.page_content.encode("utf-8")
        ).hexdigest()[:16]
    return docs


def make_splitter(path: Path, settings: Settings) -> RecursiveCharacterTextSplitter:
    code_exts = {".py", ".go", ".scala", ".sh", ".tf"}
    separators = CODE_SEPARATORS if path.suffix.lower() in code_exts else PROSE_SEPARATORS
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=separators,
    )


def process_file(path: Path, settings: Settings) -> list[Document]:
    docs = load_document(path)
    splitter = make_splitter(path, settings)
    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        chunk.metadata["source"] = str(path)
    return chunks


def dedupe_chunks(chunks: list[Document]) -> list[Document]:
    seen: set[str] = set()
    unique: list[Document] = []
    for chunk in chunks:
        h = chunk.metadata.get("content_hash") or hashlib.sha256(
            chunk.page_content.encode("utf-8")
        ).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        unique.append(chunk)
    return unique


def ingest(
    docs_dir: Path | None = None,
    persist_dir: Path | None = None,
    settings: Settings | None = None,
    max_workers: int = 8,
) -> Chroma:
    settings = settings or get_settings()
    docs_dir = docs_dir or settings.docs_dir
    persist_dir = persist_dir or settings.chroma_persist_dir
    persist_dir.mkdir(parents=True, exist_ok=True)

    files = discover_files(Path(docs_dir))
    logger.info("Discovered %d files under %s", len(files), docs_dir)

    all_chunks: list[Document] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(process_file, f, settings): f for f in files}
        for fut in as_completed(futures):
            path = futures[fut]
            try:
                all_chunks.extend(fut.result())
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to process %s: %s", path, exc)

    all_chunks = dedupe_chunks(all_chunks)
    logger.info("Embedding %d unique chunks into %s", len(all_chunks), persist_dir)

    embeddings = get_embeddings(settings)
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
        collection_name=settings.chroma_collection,
        collection_metadata={"hnsw:space": "cosine"},
    )
    return vectorstore


def load_vectorstore(settings: Settings | None = None) -> Chroma:
    settings = settings or get_settings()
    embeddings = get_embeddings(settings)
    return Chroma(
        persist_directory=str(settings.chroma_persist_dir),
        embedding_function=embeddings,
        collection_name=settings.chroma_collection,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    vs = ingest(settings=settings)
    count = vs._collection.count()  # noqa: SLF001
    print(f"Ingested into {settings.chroma_persist_dir} — collection size={count}")


if __name__ == "__main__":
    main()
