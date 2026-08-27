"""Text search — same contract as /scan, without the perception stages."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..auth import AuthUser, get_current_user_opt
from ..deps import get_state
from .scan import record_scan

router = APIRouter()


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=300)
    explain: bool = True


@router.post("/search")
async def search(
    request: SearchRequest,
    user: AuthUser | None = Depends(get_current_user_opt),
) -> dict:
    """Medicine name or composition -> the same grounded contract as /scan.

    Deliberately identical in shape. The frontend renders one result view, and
    the abstention logic applies equally: a typed query that matches nothing
    gets a refusal, not a nearest guess.
    """
    state = get_state()
    if not state.ready:
        raise HTTPException(
            status_code=503,
            detail={"error": "service not ready", "problems": state.startup_errors},
        )

    result = state.orchestrator.analyse_text(request.query, explain=request.explain)
    payload = result.to_dict()
    await record_scan(user, "search", request.query, payload)
    return payload


@router.get("/suggest")
def suggest(q: str, limit: int = 8) -> dict:
    """Type-ahead over brand names. Lexical only — no calibration, no claims."""
    state = get_state()
    if not state.ready:
        raise HTTPException(status_code=503, detail="service not ready")
    if len(q.strip()) < 2:
        return {"query": q, "suggestions": []}

    records = state.index.search(q, top_k=min(limit, 20), min_similarity=0.15)
    return {
        "query": q,
        "suggestions": [
            {
                "name": r.name,
                "composition": r.composition,
                "manufacturer": r.manufacturer,
                "similarity": round(r.similarity, 3),
            }
            for r in records
        ],
    }
