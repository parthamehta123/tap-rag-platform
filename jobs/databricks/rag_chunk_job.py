"""
Incremental doc embedding job for Databricks.

Discovers changed engineering docs (git diff / S3 manifest), chunks them,
and writes embedding payloads for the RAG ingest worker (or Bedrock Titan batch).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType


CHUNK_SIZE = 600
CHUNK_OVERLAP = 100


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def docs_to_rows(docs_dir: str) -> list[dict]:
    rows = []
    root = Path(docs_dir)
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".md", ".py", ".tf", ".yaml", ".yml"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for i, chunk in enumerate(chunk_text(text)):
            rows.append(
                {
                    "source": str(path),
                    "chunk_id": f"{path.name}:{i}",
                    "content": chunk,
                    "content_hash": hashlib.sha256(chunk.encode()).hexdigest()[:16],
                }
            )
    return rows


def main(docs_dir: str = "/dbfs/FileStore/tap-docs", output_path: str = "/dbfs/FileStore/tap-chunks") -> None:
    spark = SparkSession.builder.appName("tap-rag-chunk").getOrCreate()
    rows = docs_to_rows(docs_dir)
    schema = StructType(
        [
            StructField("source", StringType()),
            StructField("chunk_id", StringType()),
            StructField("content", StringType()),
            StructField("content_hash", StringType()),
        ]
    )
    df = spark.createDataFrame(rows, schema=schema)
    df = df.dropDuplicates(["content_hash"])
    df.write.mode("overwrite").json(output_path)
    print(json.dumps({"chunks": df.count(), "output": output_path}))
    spark.stop()


if __name__ == "__main__":
    main()
