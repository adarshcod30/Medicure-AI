"""
Authentication dependencies.

Two flavours, used deliberately:

- `get_current_user`     — the route needs an identity (history, cabinet).
- `get_current_user_opt` — the route works anonymously and merely does more
                           when an identity is present (scan, search persist
                           history for signed-in users and skip it otherwise).

Anonymous scanning is a feature, not an oversight: someone standing in a
pharmacy comparing prices should not have to create an account first.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..config import get_settings
from .security import create_token, decode_token, hash_password, verify_password

__all__ = [
    "AuthUser",
    "get_current_user",
    "get_current_user_opt",
    "create_token",
    "decode_token",
    "hash_password",
    "verify_password",
]

_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class AuthUser:
    id: str
    email: str


def _user_from_token(token: str | None) -> AuthUser | None:
    if not token:
        return None
    settings = get_settings()
    claims = decode_token(token, secret=settings.jwt_secret, algorithm=settings.jwt_algorithm)
    if not claims or "sub" not in claims:
        return None
    return AuthUser(id=str(claims["sub"]), email=str(claims.get("email", "")))


def get_current_user_opt(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> AuthUser | None:
    return _user_from_token(credentials.credentials if credentials else None)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> AuthUser:
    user = _user_from_token(credentials.credentials if credentials else None)
    if user is None:
        raise HTTPException(status_code=401, detail="not authenticated")
    return user
