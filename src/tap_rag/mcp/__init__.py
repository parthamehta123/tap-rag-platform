from tap_rag.mcp.store import FileReputationStore, HttpReputationStore, get_reputation_store

__all__ = ["FileReputationStore", "HttpReputationStore", "get_reputation_store", "mcp"]


def __getattr__(name: str):
    if name == "mcp":
        from tap_rag.mcp.server import mcp

        return mcp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
