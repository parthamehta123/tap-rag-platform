"""MCP server exposing TAP hash/IP reputation tools for LLM agents."""

from __future__ import annotations

from datetime import datetime

from mcp.server.fastmcp import FastMCP
from pydantic import ValidationError

from tap_rag.mcp.store import get_reputation_store
from tap_rag.models.schemas import (
    BulkHashRequest,
    HashReputationRequest,
    HashReputationResult,
    IPReputationRequest,
    IPReputationResult,
    Verdict,
)

mcp = FastMCP("TAP Reputation Server")


def _hash_entry(hash_value: str) -> dict | None:
    return get_reputation_store().get_hash(hash_value)


def _ip_entry(ip_address: str) -> dict | None:
    return get_reputation_store().get_ip(ip_address)


@mcp.tool()
def hash_reputation(
    hash_value: str,
    first_seen_after: str | None = None,
    last_seen_before: str | None = None,
) -> dict:
    """Look up threat reputation for a file hash (SHA256 or MD5)."""
    try:
        req = HashReputationRequest(
            hash_value=hash_value,
            first_seen_after=first_seen_after,
            last_seen_before=last_seen_before,
        )
    except ValidationError as exc:
        return {"error": str(exc)}

    entry = _hash_entry(req.hash_value)
    if entry and entry.get("error"):
        return entry
    if not entry:
        return {"error": f"Hash {req.hash_value} not found in TAP dataset"}
    if req.first_seen_after and entry.get("first_seen", "") < req.first_seen_after:
        return {"error": "Hash exists but does not match date filter"}
    if req.last_seen_before and entry.get("last_seen", "") > req.last_seen_before:
        return {"error": "Hash exists but does not match date filter"}

    result = HashReputationResult(
        hash=req.hash_value,
        verdict=Verdict(entry["verdict"]),
        threat_score=entry["threat_score"],
        first_seen=entry["first_seen"],
        last_seen=entry["last_seen"],
        prevalence=entry["prevalence"],
        family=entry.get("family"),
    )
    return result.model_dump()


@mcp.tool()
def ip_reputation(ip_address: str) -> dict:
    """Look up threat reputation for an IP address."""
    try:
        req = IPReputationRequest(ip_address=ip_address)
    except ValidationError as exc:
        return {"error": str(exc)}

    entry = _ip_entry(req.ip_address)
    if entry and entry.get("error"):
        return entry
    if not entry:
        return {"error": f"IP {req.ip_address} not found in TAP dataset"}

    result = IPReputationResult(
        ip=req.ip_address,
        verdict=Verdict(entry["verdict"]),
        threat_score=entry["threat_score"],
        country=entry["country"],
        asn=entry["asn"],
        org=entry["org"],
        associated_campaigns=entry.get("associated_campaigns", []),
    )
    return result.model_dump()


@mcp.tool()
def bulk_hash_reputation(hash_values: list[str]) -> list[dict]:
    """Look up threat reputation for multiple file hashes (max 100)."""
    try:
        req = BulkHashRequest(hash_values=hash_values)
    except ValidationError as exc:
        return [{"error": str(exc)}]

    results = []
    for h in req.hash_values:
        entry = _hash_entry(h.lower())
        if entry and not entry.get("error"):
            results.append({"hash": h.lower(), **entry})
        elif entry and entry.get("error"):
            results.append({"hash": h, **entry})
        else:
            results.append({"hash": h, "error": "not found"})
    return results


@mcp.resource("tap://datasets/summary")
def dataset_summary() -> str:
    """Summary of available TAP datasets and their coverage."""
    store = get_reputation_store()
    hash_n, ip_n = store.hash_count(), store.ip_count()
    hash_label = "remote" if hash_n < 0 else str(hash_n)
    ip_label = "remote" if ip_n < 0 else str(ip_n)
    return (
        f"TAP Reputation Datasets\n"
        f"=======================\n"
        f"Hash DB: {hash_label} entries\n"
        f"IP DB:   {ip_label} entries\n"
        f"Last updated: {datetime.now().isoformat()}\n"
        f"Supported lookups: hash_reputation, ip_reputation, bulk_hash_reputation"
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
