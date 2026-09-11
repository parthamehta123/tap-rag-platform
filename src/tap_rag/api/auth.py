"""Request authentication — Bearer shared secret or Cognito JWT."""

from __future__ import annotations

import logging
import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from tap_rag.config import Settings, get_settings

logger = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=False)

# Probes and scrapes must work without a token (ALB, kubelet, Prometheus).
PUBLIC_PATHS = frozenset({"/health", "/metrics"})


def _verify_cognito_jwt(token: str, settings: Settings) -> None:
    try:
        import jwt
    except ImportError as exc:  # pragma: no cover
        raise HTTPException(status_code=503, detail="PyJWT is required for Cognito auth") from exc

    region = settings.aws_region
    pool = settings.cognito_user_pool_id
    issuer = f"https://cognito-idp.{region}.amazonaws.com/{pool}"
    jwks_url = f"{issuer}/.well-known/jwks.json"
    try:
        signing_key = jwt.PyJWKClient(jwks_url).get_signing_key_from_jwt(token)
        decode_kwargs: dict = {
            "algorithms": ["RS256"],
            "issuer": issuer,
        }
        if settings.cognito_client_id:
            decode_kwargs["audience"] = settings.cognito_client_id
        jwt.decode(token, signing_key.key, **decode_kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.info("Cognito JWT rejected: %s", exc)
        raise HTTPException(status_code=401, detail="Invalid token") from exc


def require_auth(
    request: Request,
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    """Open locally; production fails closed unless API_AUTH_TOKEN or Cognito is set."""
    if request.url.path in PUBLIC_PATHS:
        return
    if settings.is_production and not settings.auth_configured:
        raise HTTPException(
            status_code=503,
            detail="API_AUTH_TOKEN or COGNITO_USER_POOL_ID is required in production",
        )
    if not settings.auth_required:
        return
    token = creds.credentials if creds else None
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    expected = settings.api_auth_token
    if expected and len(token) == len(expected) and secrets.compare_digest(token, expected):
        return
    if settings.cognito_user_pool_id:
        _verify_cognito_jwt(token, settings)
        return
    raise HTTPException(status_code=401, detail="Invalid token")
