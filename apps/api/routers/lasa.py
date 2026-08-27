"""
Look-alike / sound-alike confusability.

Retrieval-derived and model-free, like everything in `pharmacology/`. The
caution text is a fixed template; only the list of retrieved products varies.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from packages.pharmacology.lasa import find_confusable

from ..deps import get_state

router = APIRouter()


@router.get("/lasa")
def lasa(
    name: str = Query(min_length=2, max_length=120),
    limit: int = Query(default=10, ge=1, le=25),
) -> dict:
    """Products whose names are confusable with `name` but differ in composition.

    An empty list is the ordinary result. Most brand names have no close
    neighbour, and reporting that plainly is more useful than padding the
    response with weak matches.
    """
    state = get_state()
    if not state.ready:
        raise HTTPException(
            status_code=503,
            detail={"error": "service not ready", "problems": state.startup_errors},
        )

    # Anchor on the catalogue row when the name resolves to one, so the
    # composition exclusion has a real signature to compare against. An
    # unrecognised name still gets a purely lexical comparison.
    matches = state.index.search(name, top_k=1)
    anchor = matches[0] if matches else None

    result = find_confusable(
        state.index,
        name=anchor.name if anchor else name,
        signature=anchor.signature if anchor else (),
        exclude_row=anchor.row if anchor else None,
        limit=limit,
    )

    payload = result.to_dict()
    payload["resolved_to"] = (
        {
            "name": anchor.name,
            "composition": anchor.composition,
            "similarity": round(anchor.similarity, 3),
        }
        if anchor
        else None
    )
    if anchor is None:
        payload["message"] = (
            f"'{name}' does not match any product in the catalogue, so the "
            "comparison was made on the name alone and no composition could be "
            "excluded. " + payload["message"]
        )
    return payload
