"""
Per-user scan history.

Reading a record back returns exactly the result that was stored — the same
provenance, the same abstention, the same disclaimer. History is a filing
cabinet, not a second opinion: nothing is recomputed on read, because a
recomputation against a rebuilt index could silently disagree with what the
user was originally told.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query

from ..auth import AuthUser, get_current_user
from ..deps import get_state
from .auth import STORAGE_DISABLED, _object_id

router = APIRouter()


def _store():
    state = get_state()
    if state.store is None or not state.store.available:
        raise HTTPException(status_code=503, detail=STORAGE_DISABLED)
    return state.store


@router.get("/history")
async def list_history(
    user: AuthUser = Depends(get_current_user),
    limit: int = Query(default=20, ge=1, le=100),
    before: datetime | None = None,
) -> dict:
    """The caller's scans, newest first.

    Paginated by `created_at` rather than by offset: an offset shifts under
    inserts, so a user scrolling while scanning would see duplicates.
    """
    store = _store()
    criteria: dict = {"user_id": user.id}
    if before is not None:
        criteria["created_at"] = {"$lt": before}

    cursor = store.scans.find(criteria).sort("created_at", -1).limit(limit)
    items = []
    async for document in cursor:
        identification = (document.get("result") or {}).get("identification") or {}
        items.append(
            {
                "id": str(document["_id"]),
                "kind": document.get("kind"),
                "query": document.get("query"),
                "status": identification.get("status"),
                "composition": identification.get("composition"),
                "created_at": document.get("created_at"),
            }
        )
    return {"items": items, "count": len(items)}


@router.get("/history/{item_id}")
async def get_history_item(
    item_id: str, user: AuthUser = Depends(get_current_user)
) -> dict:
    store = _store()
    document = await store.scans.find_one(
        {"_id": _object_id(item_id), "user_id": user.id}
    )
    if document is None:
        raise HTTPException(status_code=404, detail="no such history item")
    return {
        "id": str(document["_id"]),
        "kind": document.get("kind"),
        "query": document.get("query"),
        "created_at": document.get("created_at"),
        "result": document.get("result"),
    }


@router.delete("/history/{item_id}", status_code=204)
async def delete_history_item(
    item_id: str, user: AuthUser = Depends(get_current_user)
) -> None:
    """Delete one of the caller's records.

    The user_id is part of the query, not a check after the fact, so another
    user's record and a nonexistent one are indistinguishable — deleting
    someone else's scan cannot even confirm that it exists.
    """
    store = _store()
    result = await store.scans.delete_one(
        {"_id": _object_id(item_id), "user_id": user.id}
    )
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="no such history item")
