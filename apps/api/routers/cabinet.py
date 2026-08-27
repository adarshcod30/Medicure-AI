"""
The medicine cabinet: what a user is currently taking, and what that combination implies.

Storing a cabinet item stores the composition signature the resolver produced,
not a name the user typed. Identity here is the composition — the same
decision made throughout this project — so an interaction check operates on
molecules rather than on brand strings.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from packages.pharmacology.interactions import check_signatures, get_interaction_table
from packages.storage.mongo import signature_from_json, signature_to_json

from ..auth import AuthUser, get_current_user
from ..deps import get_state
from .auth import STORAGE_DISABLED, _object_id

router = APIRouter()


class CabinetItem(BaseModel):
    display_name: str = Field(min_length=1, max_length=200)
    signature: list[list] = Field(min_length=1)
    """The composition signature exactly as /v1/scan returned it. Taking the
    client's word for the *identity* is safe because the client got it from
    this service; what is never taken on trust is any claim about it."""
    source: dict = Field(default_factory=dict)


def _store():
    state = get_state()
    if state.store is None or not state.store.available:
        raise HTTPException(status_code=503, detail=STORAGE_DISABLED)
    return state.store


@router.post("/cabinet", status_code=201)
async def add_item(item: CabinetItem, user: AuthUser = Depends(get_current_user)) -> dict:
    store = _store()
    try:
        signature = signature_from_json(item.signature)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"malformed signature: {exc}") from exc

    document = {
        "user_id": user.id,
        "display_name": item.display_name.strip(),
        "signature": signature_to_json(signature),
        "source": item.source,
        "added_at": datetime.now(timezone.utc),
    }
    result = await store.cabinet.insert_one(document)
    return {
        "id": str(result.inserted_id),
        "display_name": document["display_name"],
        "signature": document["signature"],
        "source": document["source"],
        "added_at": document["added_at"],
    }


@router.get("/cabinet")
async def list_cabinet(user: AuthUser = Depends(get_current_user)) -> dict:
    """Everything in the cabinet, plus the interaction check across all of it."""
    store = _store()
    items = []
    signatures = []
    labels = []

    async for document in store.cabinet.find({"user_id": user.id}).sort("added_at", -1):
        items.append(
            {
                "id": str(document["_id"]),
                "display_name": document.get("display_name"),
                "signature": document.get("signature"),
                "source": document.get("source", {}),
                "added_at": document.get("added_at"),
            }
        )
        signatures.append(signature_from_json(document.get("signature") or []))
        labels.append(document.get("display_name") or "(unnamed)")

    return {
        "items": items,
        "count": len(items),
        "interactions": check_signatures(signatures, labels=labels),
    }


@router.get("/cabinet/interactions")
async def cabinet_interactions(user: AuthUser = Depends(get_current_user)) -> dict:
    """The interaction check alone, for clients that already hold the items."""
    store = _store()
    signatures, labels = [], []
    async for document in store.cabinet.find({"user_id": user.id}):
        signatures.append(signature_from_json(document.get("signature") or []))
        labels.append(document.get("display_name") or "(unnamed)")
    return check_signatures(signatures, labels=labels)


@router.delete("/cabinet/{item_id}", status_code=204)
async def remove_item(item_id: str, user: AuthUser = Depends(get_current_user)) -> None:
    store = _store()
    result = await store.cabinet.delete_one(
        {"_id": _object_id(item_id), "user_id": user.id}
    )
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="no such cabinet item")
