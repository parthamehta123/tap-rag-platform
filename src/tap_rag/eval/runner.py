"""Golden-dataset evaluation pipeline for RAG (CI + online sampling)."""

from __future__ import annotations

import json
import logging
import statistics
import time
from pathlib import Path

from tap_rag.config import get_settings
from tap_rag.models.schemas import EvalMetrics, GoldenExample, RAGQuery
from tap_rag.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

# Default gates for CI (On-Demand Evaluations)
DEFAULT_ANSWER_HIT_RATE = 0.70
DEFAULT_CITATION_RECALL = 0.50
DEFAULT_HALLUCINATION_RATE_MAX = 0.20


def load_golden_dataset(path: Path) -> list[GoldenExample]:
    raw = json.loads(path.read_text())
    return [GoldenExample.model_validate(item) for item in raw]


def score_example(example: GoldenExample, answer: str, cited_sources: list[str]) -> dict:
    answer_l = answer.lower()
    hits = [kw for kw in example.expected_answer_contains if kw.lower() in answer_l]
    answer_hit = len(hits) / max(len(example.expected_answer_contains), 1)

    forbidden = [kw for kw in example.must_not_contain if kw.lower() in answer_l]
    hallucinated = len(forbidden) > 0

    # Citation metrics
    expected = {s.lower() for s in example.expected_sources}
    cited = {Path(s).name.lower() for s in cited_sources} | {s.lower() for s in cited_sources}
    if expected:
        tp = len([s for s in expected if any(s in c or c in s for c in cited)])
        citation_recall = tp / len(expected)
        citation_precision = tp / max(len(cited), 1) if cited else 0.0
    else:
        citation_recall = 1.0
        citation_precision = 1.0

    passed = answer_hit >= 0.5 and not hallucinated
    return {
        "id": example.id,
        "answer_hit_rate": answer_hit,
        "citation_recall": citation_recall,
        "citation_precision": citation_precision,
        "hallucinated": hallucinated,
        "passed": passed,
        "answer_preview": answer[:200],
    }


def run_eval(
    pipeline: RAGPipeline,
    golden: list[GoldenExample],
    answer_hit_threshold: float = DEFAULT_ANSWER_HIT_RATE,
    citation_recall_threshold: float = DEFAULT_CITATION_RECALL,
    hallucination_max: float = DEFAULT_HALLUCINATION_RATE_MAX,
) -> EvalMetrics:
    latencies: list[float] = []
    per_example: list[dict] = []

    for ex in golden:
        start = time.perf_counter()
        resp = pipeline.query(RAGQuery(question=ex.question))
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)
        cited = [c.source for c in resp.citations]
        scored = score_example(ex, resp.answer, cited)
        scored["latency_ms"] = latency
        per_example.append(scored)

    n = len(per_example) or 1
    answer_hit_rate = sum(r["answer_hit_rate"] for r in per_example) / n
    citation_precision = sum(r["citation_precision"] for r in per_example) / n
    citation_recall = sum(r["citation_recall"] for r in per_example) / n
    hallucination_rate = sum(1 for r in per_example if r["hallucinated"]) / n
    p50 = statistics.median(latencies) if latencies else 0.0
    p95 = (
        statistics.quantiles(latencies, n=20)[18]
        if len(latencies) >= 20
        else max(latencies) if latencies else 0.0
    )

    pass_threshold = (
        answer_hit_rate >= answer_hit_threshold
        and citation_recall >= citation_recall_threshold
        and hallucination_rate <= hallucination_max
    )

    return EvalMetrics(
        n_examples=len(golden),
        answer_hit_rate=answer_hit_rate,
        citation_precision=citation_precision,
        citation_recall=citation_recall,
        hallucination_rate=hallucination_rate,
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        pass_threshold=pass_threshold,
        per_example=per_example,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    golden_path = Path("data/golden/rag_golden.json")
    golden = load_golden_dataset(golden_path)
    pipeline = RAGPipeline.from_settings(settings, rebuild_index=True)
    metrics = run_eval(pipeline, golden)
    out = Path("data/golden/last_eval_report.json")
    out.write_text(metrics.model_dump_json(indent=2))
    print(metrics.model_dump_json(indent=2))
    if not metrics.pass_threshold:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
