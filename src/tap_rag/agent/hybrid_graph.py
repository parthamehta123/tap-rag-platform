"""Hybrid LangGraph: RAG retrieval + MCP tools + LoRA classification."""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from tap_rag.lora.classifier import classify_signal
from tap_rag.mcp.server import HASH_DB, IP_DB
from tap_rag.models.schemas import RAGQuery
from tap_rag.rag.pipeline import RAGPipeline


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


def reputation_node(state: HybridState) -> dict:
    text = str(
        next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            "",
        )
    )
    # Naive IOC extraction for demo
    result: dict = {"lookups": []}
    for h, entry in HASH_DB.items():
        if h in text.lower():
            result["lookups"].append({"hash": h, **entry})
    for ip, entry in IP_DB.items():
        if ip in text:
            result["lookups"].append({"ip": ip, **entry})
    if not result["lookups"]:
        result["lookups"].append({"info": "No known IOC found; try hash_reputation MCP tool"})
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
