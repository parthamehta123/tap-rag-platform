# Integration tests — skip unless RUN_INTEGRATION=1 and AWS creds present
import os

import pytest

pytestmark = pytest.mark.integration

skip = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="Set RUN_INTEGRATION=1 with AWS credentials to run",
)


@skip
def test_bedrock_rag_smoke():
    from tap_rag.config import Settings
    from tap_rag.models.schemas import RAGQuery
    from tap_rag.rag.pipeline import RAGPipeline

    settings = Settings(use_mock_llm=False, use_mock_embeddings=False)
    pipeline = RAGPipeline.from_settings(settings, rebuild_index=False)
    resp = pipeline.query(RAGQuery(question="How do I access the dev ClickHouse cluster?"))
    assert len(resp.answer) > 20
