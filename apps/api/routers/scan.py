"""Image scanning — the main path."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from ..auth import AuthUser, get_current_user_opt
from ..config import get_settings
from ..deps import get_state

logger = logging.getLogger(__name__)

router = APIRouter()


async def record_scan(user: AuthUser | None, kind: str, query: str, result: dict) -> None:
    """Persist a result for a signed-in user, or do nothing.

    Every failure here is swallowed after logging. Persistence is a
    convenience; the analysis has already succeeded and the caller is entitled
    to it whether or not the database accepted a copy. Letting a storage error
    surface would turn a working scan into a 500.
    """
    state = get_state()
    if user is None or state.store is None or not state.store.available:
        return
    try:
        await state.store.scans.insert_one(
            {
                "user_id": user.id,
                "kind": kind,
                "query": query,
                "result": result,
                "created_at": datetime.now(timezone.utc),
            }
        )
    except Exception as exc:  # noqa: BLE001 — never let storage break a scan
        logger.warning("could not persist %s for user %s: %s", kind, user.id, exc)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/heic"}


@router.post("/scan")
async def scan(
    file: UploadFile = File(...),
    explain: bool = True,
    user: AuthUser | None = Depends(get_current_user_opt),
) -> dict:
    """Photo of medicine packaging -> grounded, cited answer.

    A degraded photo returns `identification.status == "unreadable"` with an
    actionable retake message, HTTP 200. That is a successful analysis whose
    finding is "not enough information", not a client error — and the caller
    still gets the quality metrics that justify the refusal.

    Authentication is optional and deliberately so: someone standing in a
    pharmacy comparing prices should not have to create an account first. A
    signed-in caller additionally gets the result filed in their history.
    """
    state = get_state()
    settings = get_settings()

    if not state.ready:
        raise HTTPException(
            status_code=503,
            detail={"error": "service not ready", "problems": state.startup_errors},
        )

    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported content type {file.content_type}; send a JPEG, PNG or WebP",
        )

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(payload) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"image exceeds {settings.max_upload_bytes // (1024 * 1024)} MB",
        )

    try:
        result = state.orchestrator.analyse_image(payload, explain=explain)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"could not decode image: {exc}") from exc

    payload_dict = result.to_dict()
    await record_scan(user, "scan", file.filename or "(uploaded image)", payload_dict)
    return payload_dict
