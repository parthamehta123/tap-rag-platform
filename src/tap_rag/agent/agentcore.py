"""
In-process AgentCore-style runtime controls.

Used by SecurityAgent as the circuit breaker. Hosting the graph inside
Bedrock AgentCore microVMs is optional and configured via AgentCoreConfig.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AgentCoreConfig:
    """Infrastructure-level agent controls (circuit breakers outside the graph)."""

    max_iterations: int = 15
    timeout_seconds: int = 300
    memory_short_term: bool = True
    memory_long_term: bool = True
    otel_enabled: bool = True
    gateway_mcp_enabled: bool = True


class AgentCoreRuntimeAdapter:
    """Enforces iteration/timeout policy around LangGraph invoke()."""

    def __init__(self, config: AgentCoreConfig | None = None):
        self.config = config or AgentCoreConfig()

    def enforce_circuit_breaker(self, iteration: int) -> None:
        if iteration > self.config.max_iterations:
            raise RuntimeError(
                f"AgentCore max_iterations={self.config.max_iterations} exceeded"
            )

    def health(self) -> dict[str, Any]:
        return {
            "runtime": "in-process",
            "max_iterations": self.config.max_iterations,
            "timeout_seconds": self.config.timeout_seconds,
            "gateway_mcp": self.config.gateway_mcp_enabled,
            "otel": self.config.otel_enabled,
        }
