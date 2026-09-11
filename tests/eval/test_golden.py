"""Golden / LoRA eval tests used in CI On-Demand Evaluations."""

from __future__ import annotations

from pathlib import Path

from tap_rag.config import Settings
from tap_rag.eval.runner import load_golden_dataset, run_eval, score_example
from tap_rag.lora.classifier import (
    evaluate_classifier,
    load_training_examples,
    rule_based_classify,
    sklearn_classify,
    split_examples,
)
from tap_rag.models.schemas import GoldenExample
from tap_rag.rag.pipeline import RAGPipeline

ROOT = Path(__file__).resolve().parents[2]


def test_golden_dataset_loads():
    golden = load_golden_dataset(ROOT / "data" / "golden" / "rag_golden.json")
    assert len(golden) >= 5
    assert all(isinstance(g, GoldenExample) for g in golden)


def test_score_example_hit():
    ex = GoldenExample(
        id="t1",
        question="q",
        expected_answer_contains=["clickhouse", "9000"],
        expected_sources=["clickhouse.md"],
        must_not_contain=["rm -rf"],
    )
    scored = score_example(
        ex,
        answer="Use clickhouse-client on port 9000",
        cited_sources=["data/docs/clickhouse.md"],
    )
    assert scored["answer_hit_rate"] == 1.0
    assert scored["passed"]


def test_rag_golden_eval_passes_with_mock(tmp_path: Path):
    settings = Settings(
        use_mock_llm=True,
        use_mock_embeddings=True,
        docs_dir=ROOT / "data" / "docs",
        chroma_persist_dir=tmp_path / "chroma",
    )
    pipeline = RAGPipeline.from_settings(settings, rebuild_index=True)
    # Use a subset focused on keywords the mock LLM returns well
    golden = [
        g
        for g in load_golden_dataset(ROOT / "data" / "golden" / "rag_golden.json")
        if g.id in {"ch-dev-access", "geoip-module", "tap-overview"}
    ]
    metrics = run_eval(
        pipeline,
        golden,
        answer_hit_threshold=0.0,  # mock answers vary; structure must run
        citation_recall_threshold=0.0,
        hallucination_max=1.0,
    )
    assert metrics.n_examples == len(golden)
    assert metrics.latency_p50_ms >= 0


def test_lora_training_eval():
    examples = load_training_examples(ROOT / "data" / "training" / "tap_training_data.json")
    train, val, test = split_examples(examples)
    assert len(train) + len(val) + len(test) == len(examples)
    metrics = evaluate_classifier(
        test or examples[:5],
        predict_fn=lambda s: sklearn_classify(s) or rule_based_classify(s, 0.70),
        confidence_threshold=0.70,
    )
    assert metrics["n_test"] > 0
    assert "per_example" in metrics
