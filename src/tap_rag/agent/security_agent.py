"""LangGraph security agent with permission, sandbox, diff, and PR guardrails."""

from __future__ import annotations

import logging
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from tap_rag.config import Settings, get_settings
from tap_rag.models.schemas import ActionType, AgentRequest, AgentResponse, ValidationResult
from tap_rag.rag.llm import LLMClient, get_llm

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    action_type: Literal["read", "write", "execute"] | None
    proposed_diff: str | None
    validation_result: Literal["approved", "rejected"] | None
    rejection_reason: str | None
    iteration: int


def _keyword_action(user_text: str) -> str:
    text = user_text.lower()
    if any(w in text for w in ("delete", "modify", "change", "update", "write", "remove")):
        return "write"
    if any(w in text for w in ("run", "execute", "restart", "deploy")):
        return "execute"
    return "read"


class SecurityAgent:
    def __init__(self, llm: LLMClient | None = None, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.llm = llm or get_llm(self.settings)
        self.app = self._build_graph()

    def _reasoning_node(self, state: AgentState) -> dict:
        user_text = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            "",
        )
        system = SystemMessage(
            content=(
                "You are a TAP infrastructure assistant. Classify your intended action as "
                "'read' (viewing logs/configs), 'write' (modifying files), or 'execute' "
                "(running commands). Respond with the action type on the first line, then your plan."
            )
        )
        response = self.llm.invoke([system] + list(state["messages"]))
        keyword = _keyword_action(str(user_text))
        first_line = str(response.content).split("\n")[0].lower()
        if "write" in first_line:
            action = "write"
        elif "execute" in first_line:
            action = "execute"
        else:
            action = "read"
        if action == "read" and keyword in {"write", "execute"}:
            action = keyword
        return {
            "messages": [response],
            "action_type": action,
            "iteration": state.get("iteration", 0) + 1,
        }

    def _read_node(self, state: AgentState) -> dict:
        return {
            "messages": [
                AIMessage(content="[Read operation executed successfully — no mutations]")
            ]
        }

    def _sandbox_node(self, state: AgentState) -> dict:
        # Production: Docker --network none. Demo generates a representative diff.
        user_text = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            "",
        )
        if "terraform" in str(user_text).lower() or ".tf" in str(user_text).lower():
            diff = "+++ modules/geoip/main.tf\n- old_config = true\n+ new_config = false"
        else:
            diff = "+++ config.yaml\n- replicas: 3\n+ replicas: 2"
        return {"proposed_diff": diff}

    def _validation_node(self, state: AgentState) -> dict:
        diff = state.get("proposed_diff") or ""
        # Guardrail 1: block infra-critical files
        if "terraform" in diff.lower() or ".tf" in diff:
            return {
                "validation_result": "rejected",
                "rejection_reason": "Cannot modify Terraform files without elevated approval",
            }
        # Guardrail 2: block large deletions
        delete_count = diff.count("\n-")
        threshold = self.settings.agent_diff_delete_threshold
        if delete_count > threshold:
            return {
                "validation_result": "rejected",
                "rejection_reason": f"Diff deletes {delete_count} lines (threshold: {threshold})",
            }
        return {"validation_result": "approved"}

    def _pr_node(self, state: AgentState) -> dict:
        return {
            "messages": [
                AIMessage(content="PR #42 created. Awaiting human review before merge.")
            ]
        }

    def _block_node(self, state: AgentState) -> dict:
        reason = state.get("rejection_reason") or "Unknown"
        return {
            "messages": [
                AIMessage(content=f"Action BLOCKED: {reason}. No changes applied.")
            ]
        }

    def _route_action(self, state: AgentState) -> str:
        if state.get("iteration", 0) > self.settings.agent_max_iterations:
            return "block"
        if state["action_type"] == "read":
            return "read"
        return "sandbox"

    def _route_validation(self, state: AgentState) -> str:
        if state["validation_result"] == "approved":
            return "create_pr"
        return "block"

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("reasoning", self._reasoning_node)
        graph.add_node("read", self._read_node)
        graph.add_node("sandbox", self._sandbox_node)
        graph.add_node("validation", self._validation_node)
        graph.add_node("create_pr", self._pr_node)
        graph.add_node("block", self._block_node)

        graph.set_entry_point("reasoning")
        graph.add_conditional_edges(
            "reasoning", self._route_action, {"read": "read", "sandbox": "sandbox", "block": "block"}
        )
        graph.add_edge("read", END)
        graph.add_edge("sandbox", "validation")
        graph.add_conditional_edges(
            "validation",
            self._route_validation,
            {"create_pr": "create_pr", "block": "block"},
        )
        graph.add_edge("create_pr", END)
        graph.add_edge("block", END)
        return graph.compile()

    def run(self, request: AgentRequest) -> AgentResponse:
        result = self.app.invoke(
            {
                "messages": [HumanMessage(content=request.query)],
                "action_type": None,
                "proposed_diff": None,
                "validation_result": None,
                "rejection_reason": None,
                "iteration": 0,
            }
        )
        last = result["messages"][-1]
        content = last.content if isinstance(last, BaseMessage) else str(last)
        validation = result.get("validation_result")
        action = result.get("action_type")
        return AgentResponse(
            message=str(content),
            action_type=ActionType(action) if action else None,
            validation_result=ValidationResult(validation) if validation else None,
            rejection_reason=result.get("rejection_reason"),
            proposed_diff=result.get("proposed_diff"),
            blocked="BLOCKED" in str(content),
        )
