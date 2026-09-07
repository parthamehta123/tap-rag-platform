"""
Amazon Bedrock AgentCore integration stubs.

Maps TAP services onto AgentCore Runtime / Gateway / Memory / Observability.
Wire these when deploying beyond ECS into AgentCore microVMs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AgentCoreConfig:
    """Infrastructure-level agent controls (circuit breakers outside the agent)."""

    max_iterations: int = 15
    timeout_seconds: int = 300
    memory_short_term: bool = True
    memory_long_term: bool = True
    otel_enabled: bool = True
    gateway_mcp_enabled: bool = True


class AgentCoreRuntimeAdapter:
    """
    Adapter that would host the LangGraph security / hybrid agent inside
    Bedrock AgentCore Runtime (per-session microVM isolation).

    Production wiring:
      - AgentCore Runtime: execute LangGraph app.invoke(...)
      - AgentCore Gateway: expose MCP reputation tools
      - AgentCore Memory: replace st.session_state with managed memory
      - AgentCore Identity: Cognito → scoped IAM for tool calls
      - AgentCore Observability: OpenTelemetry traces for tool selection
      - Dual-layer eval: On-Demand (CI) + Online (sampled live traffic)
    """

    def __init__(self, config: AgentCoreConfig | None = None):
        self.config = config or AgentCoreConfig()

    def enforce_circuit_breaker(self, iteration: int) -> None:
        if iteration > self.config.max_iterations:
            raise RuntimeError(
                f"AgentCore max_iterations={self.config.max_iterations} exceeded"
            )

    def health(self) -> dict[str, Any]:
        return {
            "runtime": "bedrock-agentcore",
            "max_iterations": self.config.max_iterations,
            "timeout_seconds": self.config.timeout_seconds,
            "gateway_mcp": self.config.gateway_mcp_enabled,
            "otel": self.config.otel_enabled,
        }
