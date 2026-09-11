"""Hybrid LangGraph: RAG retrieval + MCP tools + LoRA classification."""

from __future__ import annotations

import re
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from tap_rag.lora.classifier import classify_signal
from tap_rag.mcp.store import get_reputation_store
from tap_rag.models.schemas import RAGQuery
from tap_rag.rag.pipeline import RAGPipeline

HASH_RE = re.compile(r"\b[a-fA-F0-9]{8,64}\b")
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


class HybridState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: Literal["rag", "reputation", "anomaly", "unknown"] | None
    rag_answer: str | None
    tool_result: dict | None
    classification: dict | None


def classify_intent(state: HybridState) -> dict:
    text = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    )
    lower = str(text).lower()
    if any(k in lower for k in ("hash", "ip ", "reputation", "ioc", "malicious")):
        intent: Literal["rag", "reputation", "anomaly", "unknown"] = "reputation"
    elif any(k in lower for k in ("classify", "prevalence", "anomaly", "observable")):
        intent = "anomaly"
    elif any(k in lower for k in ("how", "where", "what", "clickhouse", "terraform", "spark")):
        intent = "rag"
    else:
        intent = "rag"
    return {"intent": intent}


def rag_node(state: HybridState, pipeline: RAGPipeline) -> dict:
    text = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    )
    resp = pipeline.query(RAGQuery(question=str(text)))
    return {
        "rag_answer": resp.answer,
        "messages": [AIMessage(content=resp.answer)],
    }


def extract_iocs(text: str) -> tuple[list[str], list[str]]:
    hashes = list(dict.fromkeys(m.lower() for m in HASH_RE.findall(text)))
    ips = list(dict.fromkeys(IP_RE.findall(text)))
    return hashes, ips


def reputation_node(state: HybridState) -> dict:
    text = str(
        next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            "",
        )
    )
    store = get_reputation_store()
    hashes, ips = extract_iocs(text)
    lookups: list[dict] = []
    for h in hashes:
        entry = store.get_hash(h)
        if entry and not entry.get("error"):
            lookups.append({"hash": h, **entry})
    for ip in ips:
        entry = store.get_ip(ip)
        if entry and not entry.get("error"):
            lookups.append({"ip": ip, **entry})
    if not lookups:
        lookups.append({"info": "No IOC extracted or none found in the reputation backend"})
    result = {"lookups": lookups}
    msg = AIMessage(content=f"Reputation lookup result: {result}")
    return {"tool_result": result, "messages": [msg]}


def anomaly_node(state: HybridState) -> dict:
    text = str(
        next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            "",
        )
    )
    classification = classify_signal(text)
    payload = classification.model_dump()
    msg = AIMessage(content=f"Anomaly classification: {payload}")
    return {"classification": payload, "messages": [msg]}


def route_intent(state: HybridState) -> str:
    return state.get("intent") or "rag"


def build_hybrid_graph(pipeline: RAGPipeline):
    graph = StateGraph(HybridState)
    graph.add_node("intent", classify_intent)
    graph.add_node("rag", lambda s: rag_node(s, pipeline))
    graph.add_node("reputation", reputation_node)
    graph.add_node("anomaly", anomaly_node)
    graph.set_entry_point("intent")
    graph.add_conditional_edges(
        "intent",
        route_intent,
        {"rag": "rag", "reputation": "reputation", "anomaly": "anomaly", "unknown": "rag"},
    )
    graph.add_edge("rag", END)
    graph.add_edge("reputation", END)
    graph.add_edge("anomaly", END)
    return graph.compile()
