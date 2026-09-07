"""Streamlit CloudOps RAG chatbot UI."""

from __future__ import annotations

import os
import random

import streamlit as st

from tap_rag.config import get_settings
from tap_rag.models.schemas import FeedbackEvent, RAGQuery
from tap_rag.rag.pipeline import RAGPipeline

SAMPLE_QUESTIONS = [
    "How do I access the dev ClickHouse cluster?",
    "Where is the GeoIP Terraform module and what does it provision?",
    "How does the TAP Spark prevalence job write to S3?",
    "What Vault path stores ClickHouse credentials?",
    "How do I authenticate MCP reputation API calls?",
]


@st.cache_resource
def get_pipeline() -> RAGPipeline:
    return RAGPipeline.from_settings(get_settings())


def main() -> None:
    settings = get_settings()
    st.set_page_config(page_title="TAP RAG Chatbot", page_icon="🛡️", layout="wide")
    st.title("TAP CloudOps RAG")
    st.caption(
        f"env={settings.app_env} · mock_llm={settings.use_mock_llm} · "
        f"model={settings.bedrock_llm_model_id}"
    )

    with st.sidebar:
        st.header("Settings")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
        top_k = st.slider("Top-K retrieval", 1, 10, settings.retriever_top_k)
        if st.button("Generate random question"):
            st.session_state.pending_question = random.choice(SAMPLE_QUESTIONS)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander("Sources"):
                    for c in msg["citations"]:
                        st.markdown(f"- `{c}`")

    prompt = st.session_state.pop("pending_question", None) or st.chat_input(
        "Ask about TAP docs, Terraform, Spark, ClickHouse…"
    )
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    pipeline = get_pipeline()
    with st.chat_message("assistant"):
        response = pipeline.query(
            RAGQuery(question=prompt, top_k=top_k, temperature=temperature)
        )
        st.markdown(response.answer)
        citations = [c.source for c in response.citations]
        if citations:
            with st.expander("Sources"):
                for c in citations:
                    st.markdown(f"- `{c}`")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Helpful", key=f"up-{len(st.session_state.messages)}"):
                FeedbackEvent(
                    question=prompt, answer=response.answer, rating="up"
                ).model_dump()
                st.toast("Thanks — logged for RLHF")
        with col2:
            if st.button("👎 Not helpful", key=f"down-{len(st.session_state.messages)}"):
                st.toast("Thanks — logged for RLHF")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response.answer,
            "citations": citations,
        }
    )


if __name__ == "__main__":
    main()
