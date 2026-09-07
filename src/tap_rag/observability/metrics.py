"""Lightweight observability helpers (OpenTelemetry-ready)."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator

from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

RAG_REQUESTS = Counter("tap_rag_requests_total", "RAG queries", ["status"])
RAG_LATENCY = Histogram("tap_rag_latency_seconds", "RAG latency")
AGENT_BLOCKS = Counter("tap_agent_blocks_total", "Security agent blocks")


@contextmanager
def timed(operation: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("op=%s latency_ms=%.1f", operation, elapsed * 1000)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
