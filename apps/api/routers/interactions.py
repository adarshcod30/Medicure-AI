"""
Stateless interaction checking.

Same engine as the cabinet, without persistence — for a caller who holds the
signatures already and wants to check a hypothetical combination. Needs no
account, because it stores nothing.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from packages.pharmacology.interactions import check_signatures, get_interaction_table
from packages.storage.mongo import signature_from_json

router = APIRouter()

MAX_ITEMS = 25
"""The check is pairwise, so cost grows with the square of the item count. A
cabinet larger than this is not a realistic personal medicine list and is more
likely a client bug or an attempt to make the server do quadratic work."""


class CheckRequest(BaseModel):
    signatures: list[list[list]] = Field(min_length=1, max_length=MAX_ITEMS)
    labels: list[str] | None = None


@router.post("/interactions/check")
def check(request: CheckRequest) -> dict:
    """Check a set of composition signatures against each other.

    An empty `findings` list means nothing is on record, which is not the same
    as safe — `coverage_note` says so in every response, including this one.
    """
    if request.labels is not None and len(request.labels) != len(request.signatures):
        raise HTTPException(
            status_code=422,
            detail="labels, when given, must be the same length as signatures",
        )

    try:
        signatures = [signature_from_json(s) for s in request.signatures]
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"malformed signature: {exc}") from exc

    return check_signatures(signatures, labels=request.labels)


@router.get("/interactions/status")
def status() -> dict:
    """Whether the dataset is installed, and how much of it there is."""
    return get_interaction_table().status()
