"""Reputation backends: JSON files locally, HTTP API in production."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Protocol

from tap_rag.config import Settings, get_settings

logger = logging.getLogger(__name__)


class ReputationStore(Protocol):
    def get_hash(self, hash_value: str) -> dict | None: ...
    def get_ip(self, ip_address: str) -> dict | None: ...
    def hash_count(self) -> int: ...
    def ip_count(self) -> int: ...


class FileReputationStore:
    """Load TAP hash/IP JSON dumps from disk (replace files with a real TAP export)."""

    def __init__(self, data_dir: Path):
        self._hashes = _load_json_map(data_dir / "hashes.json")
        self._ips = _load_json_map(data_dir / "ips.json")

    def get_hash(self, hash_value: str) -> dict | None:
        return self._hashes.get(hash_value.lower())

    def get_ip(self, ip_address: str) -> dict | None:
        return self._ips.get(ip_address)

    def hash_count(self) -> int:
        return len(self._hashes)

    def ip_count(self) -> int:
        return len(self._ips)


class HttpReputationStore:
    """GET {base}/v1/hashes/{hash} and {base}/v1/ips/{ip} — TAP or equivalent API."""

    def __init__(self, base_url: str, token: str = "", timeout: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        if not self.token:
            return {}
        return {"Authorization": f"Bearer {self.token}"}

    def get_hash(self, hash_value: str) -> dict | None:
        return self._get(f"/v1/hashes/{hash_value.lower()}")

    def get_ip(self, ip_address: str) -> dict | None:
        return self._get(f"/v1/ips/{ip_address}")

    def hash_count(self) -> int:
        return -1

    def ip_count(self) -> int:
        return -1

    def _get(self, path: str) -> dict | None:
        import httpx

        url = f"{self.base_url}{path}"
        try:
            response = httpx.get(url, headers=self._headers(), timeout=self.timeout)
        except httpx.HTTPError as exc:
            logger.warning("Reputation API request failed %s: %s", url, exc)
            return {"error": "reputation backend unavailable"}
        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            logger.warning("Reputation API %s → %s", url, response.status_code)
            return {"error": f"reputation backend HTTP {response.status_code}"}
        payload = response.json()
        return payload if isinstance(payload, dict) else None


def _load_json_map(path: Path) -> dict[str, dict]:
    if not path.exists():
        logger.warning("Reputation file missing: %s", path)
        return {}
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must be a JSON object")
    return {str(k).lower(): v for k, v in raw.items()}


def get_reputation_store(settings: Settings | None = None) -> ReputationStore:
    settings = settings or get_settings()
    if settings.reputation_api_base:
        logger.info("Reputation backend: HTTP %s", settings.reputation_api_base)
        return HttpReputationStore(
            settings.reputation_api_base,
            token=settings.reputation_api_token,
        )
    data_dir = Path(settings.reputation_data_dir)
    logger.info("Reputation backend: file %s", data_dir)
    return FileReputationStore(data_dir)
