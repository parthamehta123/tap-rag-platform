"""MCP server exposing TAP hash/IP reputation tools for LLM agents."""

from __future__ import annotations

from datetime import datetime

from mcp.server.fastmcp import FastMCP
from pydantic import ValidationError

from tap_rag.models.schemas import (
    BulkHashRequest,
    HashReputationRequest,
    HashReputationResult,
    IPReputationRequest,
    IPReputationResult,
    Verdict,
)

mcp = FastMCP("TAP Reputation Server")

HASH_DB: dict[str, dict] = {
    "e3b0c44298fc1c14": {
        "verdict": "malicious",
        "threat_score": 95,
        "first_seen": "2025-03-15",
        "last_seen": "2025-10-01",
        "prevalence": 42000,
        "family": "TrojanDropper.GenericKD",
    },
    "a1b2c3d4e5f60718": {
        "verdict": "clean",
        "threat_score": 0,
        "first_seen": "2024-01-10",
        "last_seen": "2025-10-01",
        "prevalence": 12340,
        "family": None,
    },
    "deadbeef12345678": {
        "verdict": "suspicious",
        "threat_score": 88,
        "first_seen": "2026-06-01",
        "last_seen": "2026-07-10",
        "prevalence": 27000,
        "family": "Unknown.CampaignX",
    },
}

IP_DB: dict[str, dict] = {
    "10.0.5.22": {
        "verdict": "suspicious",
        "threat_score": 72,
        "country": "US",
        "asn": "AS15169",
        "org": "Internal",
        "associated_campaigns": ["APT29-Recon"],
    },
    "8.8.8.8": {
        "verdict": "clean",
        "threat_score": 0,
        "country": "US",
        "asn": "AS15169",
        "org": "Google LLC",
        "associated_campaigns": [],
    },
}


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

    entry = HASH_DB.get(req.hash_value)
    if not entry:
        return {"error": f"Hash {req.hash_value} not found in TAP dataset"}
    if req.first_seen_after and entry["first_seen"] < req.first_seen_after:
        return {"error": "Hash exists but does not match date filter"}
    if req.last_seen_before and entry["last_seen"] > req.last_seen_before:
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

    entry = IP_DB.get(req.ip_address)
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
        entry = HASH_DB.get(h.lower())
        if entry:
            results.append({"hash": h.lower(), **entry})
        else:
            results.append({"hash": h, "error": "not found"})
    return results


@mcp.resource("tap://datasets/summary")
def dataset_summary() -> str:
    """Summary of available TAP datasets and their coverage."""
    return (
        f"TAP Reputation Datasets\n"
        f"=======================\n"
        f"Hash DB: {len(HASH_DB)} entries\n"
        f"IP DB:   {len(IP_DB)} entries\n"
        f"Last updated: {datetime.now().isoformat()}\n"
        f"Supported lookups: hash_reputation, ip_reputation, bulk_hash_reputation"
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
