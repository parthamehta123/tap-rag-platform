"""Lightweight observability helpers — Prometheus metrics + OpenTelemetry tracing."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

# ── Prometheus metrics ──

RAG_REQUESTS = Counter("tap_rag_requests_total", "RAG queries", ["status"])
RAG_LATENCY = Histogram(
    "tap_rag_latency_seconds",
    "RAG end-to-end latency",
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
AGENT_BLOCKS = Counter("tap_agent_blocks_total", "Security agent blocks")
LORA_CLASSIFICATIONS = Counter(
    "tap_lora_classifications_total", "LoRA classify calls", ["label"]
)
FEEDBACK_EVENTS = Counter("tap_feedback_events_total", "Feedback events", ["rating"])


@contextmanager
def timed(operation: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield # this yields the control to the code block inside the with statement
    finally:
        elapsed = time.perf_counter() - start
        logger.info("op=%s latency_ms=%.1f", operation, elapsed * 1000)


# ── OpenTelemetry tracing ──

def init_tracing(endpoint: str, service_name: str = "tap-rag-platform") -> None:
    """Initialise OTEL tracing with OTLP exporter. No-op if endpoint is empty."""
    if not endpoint:
        logger.info("OTEL endpoint not configured — tracing disabled")
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        trace.set_tracer_provider(provider)
        logger.info("OTEL tracing initialised → %s", endpoint)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to initialise OTEL tracing: %s", exc)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
