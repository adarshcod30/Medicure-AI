"""
MongoDB access, built to be absent.

The API must serve identification, price checks and abstention with no
database at all — persistence is a convenience layered on top, not a
dependency underneath. So this wrapper follows the same contract as Bedrock
in `apps/api/deps.py`: construct cheaply, probe at startup, report honestly,
and let every caller ask `available` before relying on it.

Collections:

    users     {_id, email, name, password_hash, created_at}
    scans     {_id, user_id, kind, query, result, created_at}
    cabinet   {_id, user_id, display_name, signature, source, added_at}

`signature` is stored in the JSON-safe list form produced by
`signature_to_json` below; `signature_from_json` restores the tuple form the
resolver and pharmacology engines expect. Round-tripping through these two
functions is the only supported way in or out of storage.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Signatures are tuples of (ingredient, strength, unit, concentration) — see
# packages/resolver/normalize.py. Mongo stores lists; the engines need tuples
# back, because signatures are dict keys throughout the pharmacology package.


def signature_to_json(signature: tuple) -> list:
    return [list(component) for component in signature]


def signature_from_json(payload: list) -> tuple:
    return tuple(tuple(component) for component in payload)


class MongoStore:
    """Motor client with an explicit availability contract.

    `connect()` is the only place that talks to the network at startup, and it
    is bounded: a 3-second server-selection timeout, one ping. A laptop with
    no `MONGODB_URI` configured pays three seconds once and then runs exactly
    as before this module existed.
    """

    PING_TIMEOUT_MS = 3000

    def __init__(self, uri: str, database: str) -> None:
        self.uri = uri
        self.database_name = database
        self.available = False
        self.last_error: str | None = None
        self._client: Any = None
        self._db: Any = None

    async def connect(self) -> bool:
        try:
            from motor.motor_asyncio import AsyncIOMotorClient

            self._client = AsyncIOMotorClient(
                self.uri, serverSelectionTimeoutMS=self.PING_TIMEOUT_MS
            )
            await self._client.admin.command("ping")
            self._db = self._client[self.database_name]
            self.available = True
            self.last_error = None
            await self._ensure_indexes()
            logger.info("mongo connected: database=%s", self.database_name)
        except Exception as exc:  # noqa: BLE001 — any failure means "no storage", never "no service"
            self.available = False
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("mongo unavailable, continuing without persistence: %s", exc)
        return self.available

    async def _ensure_indexes(self) -> None:
        await self._db.users.create_index("email", unique=True)
        await self._db.scans.create_index([("user_id", 1), ("created_at", -1)])
        await self._db.cabinet.create_index([("user_id", 1), ("added_at", -1)])

    def close(self) -> None:
        if self._client is not None:
            self._client.close()

    @property
    def db(self) -> Any:
        if not self.available:
            raise RuntimeError(
                "MongoDB is not available"
                + (f" ({self.last_error})" if self.last_error else "")
            )
        return self._db

    @property
    def users(self) -> Any:
        return self.db.users

    @property
    def scans(self) -> Any:
        return self.db.scans

    @property
    def cabinet(self) -> Any:
        return self.db.cabinet

    def status(self) -> dict:
        """For /v1/health — state the capability, never imply it."""
        return {
            "available": self.available,
            "database": self.database_name,
            "error": self.last_error,
        }
