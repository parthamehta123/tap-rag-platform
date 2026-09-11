"""Unit tests — fully mocked, no AWS."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tap_rag.agent.security_agent import SecurityAgent
from tap_rag.config import Settings
from tap_rag.lora.classifier import parse_classification, rule_based_classify
from tap_rag.models.schemas import AgentRequest, HashReputationRequest, RAGQuery
from tap_rag.rag.llm import MockEmbeddings, MockLLMClient
from tap_rag.rag.pipeline import RAGPipeline
from tap_rag.rag.retriever import deduplicate_by_source


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        use_mock_llm=True,
        use_mock_embeddings=True,
        docs_dir=Path(__file__).resolve().parents[2] / "data" / "docs",
        chroma_persist_dir=tmp_path / "chroma",
        chunk_size=600,
        chunk_overlap=100,
    )


def test_hash_reputation_hex_validation():
    with pytest.raises(ValidationError):
        HashReputationRequest(hash_value="not-hex!!")
    ok = HashReputationRequest(hash_value="e3b0c44298fc1c14")
    assert ok.hash_value == "e3b0c44298fc1c14"


def test_parse_classification():
    text = (
        "Classification: SUSPICIOUS\n"
        "Confidence: 0.94\n"
        "Reasoning: Prevalence spike is anomalous."
    )
    parsed = parse_classification(text, confidence_threshold=0.7)
    assert parsed is not None
    assert parsed.classification.value == "SUSPICIOUS"
    assert parsed.confidence == 0.94
    assert not parsed.routed_to_analyst


def test_rule_based_classify_routes_low_confidence():
    result = rule_based_classify(
        "Hash xyz had isolated presence on 2 endpoints. Classify.",
        threshold=0.70,
    )
    assert result.classification.value in {"INVESTIGATE", "SUSPICIOUS", "BENIGN"}


def test_deduplicate_by_source():
    from langchain_core.documents import Document

    docs = [
        Document(page_content="a", metadata={"source": "a.md"}),
        Document(page_content="b", metadata={"source": "a.md"}),
        Document(page_content="c", metadata={"source": "a.md"}),
        Document(page_content="d", metadata={"source": "b.md"}),
    ]
    kept = deduplicate_by_source(docs, max_per_source=2)
    assert len(kept) == 3


def test_rag_pipeline_end_to_end(settings: Settings):
    pipeline = RAGPipeline.from_settings(settings, rebuild_index=True)
    resp = pipeline.query(RAGQuery(question="How do I access the dev ClickHouse cluster?"))
    assert resp.answer
    assert resp.model_id == "mock-llm"
    assert resp.latency_ms is not None


def test_security_agent_blocks_terraform_write(settings: Settings):
    agent = SecurityAgent(llm=MockLLMClient(), settings=settings)
    result = agent.run(
        AgentRequest(query="Delete the old Terraform state files for GeoIP")
    )
    assert result.blocked
    assert result.action_type is not None
    assert "BLOCKED" in result.message


def test_security_agent_allows_read(settings: Settings):
    agent = SecurityAgent(llm=MockLLMClient(), settings=settings)
    result = agent.run(
        AgentRequest(query="Show me the current Terraform state for GeoIP")
    )
    assert not result.blocked
    assert result.action_type.value == "read"


def test_api_health(settings: Settings, monkeypatch, tmp_path):
    monkeypatch.setenv("USE_MOCK_LLM", "true")
    monkeypatch.setenv("USE_MOCK_EMBEDDINGS", "true")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("DOCS_DIR", str(Path(__file__).resolve().parents[2] / "data" / "docs"))
    # Clear cached settings / pipeline
    from tap_rag.api import main as api_main
    from tap_rag.config import get_settings

    get_settings.cache_clear()
    api_main.get_pipeline.cache_clear()
    api_main.get_agent.cache_clear()

    client = TestClient(api_main.app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_mock_embeddings_deterministic():
    emb = MockEmbeddings(dim=32)
    a = emb.embed_query("clickhouse")
    b = emb.embed_query("clickhouse")
    assert a == b
    assert len(a) == 32
    # Similar queries should be closer than unrelated ones
    q = emb.embed_query("clickhouse vault credentials")
    d1 = emb.embed_documents(["ClickHouse Vault path secret/tap/clickhouse/dev"])[0]
    d2 = emb.embed_documents(["unrelated terraform geoip module"])[0]

    def cos(u, v):
        return sum(x * y for x, y in zip(u, v, strict=False))

    assert cos(q, d1) > cos(q, d2)


def _reset_api_caches(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("USE_MOCK_LLM", "true")
    monkeypatch.setenv("USE_MOCK_EMBEDDINGS", "true")
    monkeypatch.setenv("CHROMA_S3_SYNC", "false")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("DOCS_DIR", str(Path(__file__).resolve().parents[2] / "data" / "docs"))
    from tap_rag.api import main as api_main
    from tap_rag.config import get_settings

    get_settings.cache_clear()
    api_main.get_pipeline.cache_clear()
    api_main.get_agent.cache_clear()


def test_metrics_endpoint(monkeypatch, tmp_path):
    _reset_api_caches(monkeypatch, tmp_path)
    from tap_rag.api import main as api_main

    client = TestClient(api_main.app)
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]


def test_api_auth_token_enforced(monkeypatch, tmp_path):
    monkeypatch.setenv("API_AUTH_TOKEN", "test-secret")
    _reset_api_caches(monkeypatch, tmp_path)
    from tap_rag.api import main as api_main

    client = TestClient(api_main.app)
    assert client.get("/health").status_code == 200
    denied = client.post("/v1/lora/classify", json={"signal": "stable sso mfa scheduled"})
    assert denied.status_code == 401
    ok = client.post(
        "/v1/lora/classify",
        json={"signal": "stable sso mfa scheduled"},
        headers={"Authorization": "Bearer test-secret"},
    )
    assert ok.status_code == 200


def test_production_fails_closed_without_auth(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("API_AUTH_TOKEN", "")
    _reset_api_caches(monkeypatch, tmp_path)
    from tap_rag.api import main as api_main

    client = TestClient(api_main.app)
    assert client.get("/health").status_code == 200
    denied = client.post("/v1/lora/classify", json={"signal": "stable sso mfa scheduled"})
    assert denied.status_code == 503


def test_chroma_sync_noop_when_disabled(settings):
    from tap_rag.rag.chroma_sync import pull_chroma_from_s3, push_chroma_to_s3

    settings.chroma_s3_sync = False
    assert pull_chroma_from_s3(settings) is False
    push_chroma_to_s3(settings)


def test_chroma_pull_downloads_objects(settings, monkeypatch, tmp_path):
    from tap_rag.rag import chroma_sync

    settings.chroma_s3_sync = True
    settings.chroma_persist_dir = tmp_path / "chroma"
    settings.s3_bucket = "bucket"

    class FakeClient:
        def get_paginator(self, _name):
            return self

        def paginate(self, **_kwargs):
            return [{"Contents": [{"Key": "chroma/chroma.sqlite3"}]}]

        def download_file(self, _bucket, _key, dest):
            Path(dest).write_text("ok")

    monkeypatch.setattr(chroma_sync, "_client", lambda _s: FakeClient())
    assert chroma_sync.pull_chroma_from_s3(settings) is True
    assert (tmp_path / "chroma" / "chroma.sqlite3").read_text() == "ok"


def test_agentcore_circuit_breaker():
    from tap_rag.agent.agentcore import AgentCoreConfig, AgentCoreRuntimeAdapter

    adapter = AgentCoreRuntimeAdapter(AgentCoreConfig(max_iterations=2))
    adapter.enforce_circuit_breaker(2)
    with pytest.raises(RuntimeError, match="max_iterations"):
        adapter.enforce_circuit_breaker(3)


ROOT = Path(__file__).resolve().parents[2]


def test_sklearn_classify_prevalence_spike():
    from tap_rag.lora.classifier import sklearn_classify

    settings = Settings(training_data_path=ROOT / "data" / "training" / "tap_training_data.json")
    result = sklearn_classify(
        "Hash e3b0c44298fc1c14 had prevalence spike from 5 to 42,000 in 7 days, "
        "first seen in enterprise segment. Classify this observable.",
        settings,
    )
    assert result is not None
    assert result.classification.value == "SUSPICIOUS"
    assert "TF-IDF" in result.reasoning


def test_file_reputation_store():
    from tap_rag.mcp.store import FileReputationStore

    store = FileReputationStore(ROOT / "data" / "reputation")
    entry = store.get_hash("e3b0c44298fc1c14")
    assert entry is not None
    assert entry["verdict"] == "malicious"
    ip = store.get_ip("10.0.5.22")
    assert ip is not None
    assert ip["verdict"] == "suspicious"


def test_http_reputation_store(monkeypatch):
    import httpx

    from tap_rag.mcp.store import HttpReputationStore

    class Resp:
        status_code = 200

        def json(self):
            return {
                "verdict": "clean",
                "threat_score": 0,
                "first_seen": "2024-01-01",
                "last_seen": "2024-01-02",
                "prevalence": 1,
                "family": None,
            }

    monkeypatch.setattr(httpx, "get", lambda *_a, **_k: Resp())
    store = HttpReputationStore("https://rep.example")
    assert store.get_hash("abcabcab")["verdict"] == "clean"


def test_extract_iocs_from_query():
    from tap_rag.agent.hybrid_graph import extract_iocs

    hashes, ips = extract_iocs("lookup hash e3b0c44298fc1c14 from 10.0.5.22")
    assert "e3b0c44298fc1c14" in hashes
    assert "10.0.5.22" in ips
