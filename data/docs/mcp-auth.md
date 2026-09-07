# MCP Reputation API Authentication

## Auth — AWS SigV4

Every reputation API call is cryptographically signed with temporary IAM credentials
(SigV4). Do not store static API keys.

## Available Tools (MCP)

- `hash_reputation` — SHA256/MD5 threat lookup
- `ip_reputation` — IP geolocation + threat score
- `bulk_hash_reputation` — batch up to 100 hashes

## Timestamp Filters

Use `first_seen_after` and `last_seen_before` ISO date params to scope lookups
to an incident window.

## Clients

Claude, Amazon Q, VS Code extensions, and LangGraph agents all discover tools
via the MCP protocol — build once, connect everywhere.
