"""Text search — same contract as /scan, without the perception stages."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..deps import get_state

router = APIRouter()


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=300)
    explain: bool = True


@router.post("/search")
def search(request: SearchRequest) -> dict:
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
    return result.to_dict()


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
