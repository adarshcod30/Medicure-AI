"""
Accounts: register, login, me.

Storage is optional across this whole service, so every route here begins by
checking for it and returns 503 with a plain explanation when it is absent.
That is deliberately different from 500: running without MongoDB is a
supported configuration in which scanning, pricing and abstention all work and
only accounts are unavailable. The message says exactly that, because an
operator reading "internal server error" learns nothing.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr, Field

from ..auth import AuthUser, create_token, get_current_user, hash_password, verify_password
from ..config import get_settings
from ..deps import get_state

logger = logging.getLogger(__name__)

router = APIRouter()


def _object_id(value: str):
    """Mongo's _id, or a 404-safe sentinel for a malformed one.

    A token carrying a non-ObjectId subject must not raise a 500 out of the
    driver — it is simply an id that matches nothing.
    """
    from bson import ObjectId
    from bson.errors import InvalidId

    try:
        return ObjectId(value)
    except (InvalidId, TypeError):
        return ObjectId("0" * 24)

STORAGE_DISABLED = (
    "Accounts are disabled on this deployment because no database is "
    "configured. Scanning, price checks and alternatives are unaffected."
)


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=72)
    name: str = Field(default="", max_length=120)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=72)


def _store():
    state = get_state()
    if state.store is None or not state.store.available:
        raise HTTPException(status_code=503, detail=STORAGE_DISABLED)
    return state.store


def _issue(user_id: str, email: str) -> str:
    settings = get_settings()
    return create_token(
        subject=user_id,
        email=email,
        secret=settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
        expiry_minutes=settings.jwt_expiry_minutes,
    )


@router.post("/register", status_code=201)
async def register(request: RegisterRequest) -> dict:
    store = _store()
    email = request.email.lower().strip()

    document = {
        "email": email,
        "name": request.name.strip(),
        "password_hash": hash_password(request.password),
        "created_at": datetime.now(timezone.utc),
    }

    try:
        # Rely on the unique index rather than a find-then-insert check, which
        # races: two simultaneous registrations both see "no such user" and
        # both proceed. The database is the only place that can decide this.
        result = await store.users.insert_one(document)
    except Exception as exc:  # noqa: BLE001
        if type(exc).__name__ == "DuplicateKeyError":
            raise HTTPException(status_code=409, detail="that email is already registered") from exc
        raise

    user_id = str(result.inserted_id)
    return {
        "token": _issue(user_id, email),
        "user": {"id": user_id, "email": email, "name": document["name"]},
    }


@router.post("/login")
async def login(request: LoginRequest) -> dict:
    store = _store()
    email = request.email.lower().strip()
    user = await store.users.find_one({"email": email})

    # One message and one status for both "no such account" and "wrong
    # password". Distinguishing them turns this endpoint into an oracle for
    # which addresses are registered.
    if user is None or not verify_password(request.password, user.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="incorrect email or password")

    user_id = str(user["_id"])
    return {
        "token": _issue(user_id, email),
        "user": {"id": user_id, "email": email, "name": user.get("name", "")},
    }


@router.get("/me")
async def me(user: AuthUser = Depends(get_current_user)) -> dict:
    """The signed-in user's own record, never the password hash."""
    store = _store()
    document = await store.users.find_one({"_id": _object_id(user.id)})
    if document is None:
        # A token that decodes but names nobody: the account was deleted while
        # the token was still inside its expiry window.
        raise HTTPException(status_code=401, detail="account no longer exists")
    return {
        "id": str(document["_id"]),
        "email": document["email"],
        "name": document.get("name", ""),
        "created_at": document.get("created_at"),
    }
