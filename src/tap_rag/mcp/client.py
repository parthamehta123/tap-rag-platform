"""MCP client helper for discovering and calling TAP reputation tools."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_SCRIPT = Path(__file__).resolve().parent / "server.py"


def format_tool_result(result) -> str:
    if result.structuredContent is not None:
        return json.dumps(result.structuredContent, indent=2)
    if result.content:
        parts = []
        for block in result.content:
            if hasattr(block, "text"):
                parts.append(block.text)
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(result)


async def demo() -> None:
    server = StdioServerParameters(command=sys.executable, args=[str(SERVER_SCRIPT)])
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("=== Available tools ===")
            for tool in tools.tools:
                print(f"  {tool.name}: {tool.description}")

            result = await session.call_tool(
                "hash_reputation", {"hash_value": "e3b0c44298fc1c14"}
            )
            print("\n=== hash_reputation ===")
            print(format_tool_result(result))

            result = await session.call_tool(
                "ip_reputation", {"ip_address": "10.0.5.22"}
            )
            print("\n=== ip_reputation ===")
            print(format_tool_result(result))


def main() -> None:
    asyncio.run(demo())


if __name__ == "__main__":
    main()
