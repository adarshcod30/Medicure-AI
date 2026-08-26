"""Image scanning — the main path."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from ..config import get_settings
from ..deps import get_state

router = APIRouter()

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/heic"}


@router.post("/scan")
async def scan(file: UploadFile = File(...), explain: bool = True) -> dict:
    """Photo of medicine packaging -> grounded, cited answer.

    A degraded photo returns `identification.status == "unreadable"` with an
    actionable retake message, HTTP 200. That is a successful analysis whose
    finding is "not enough information", not a client error — and the caller
    still gets the quality metrics that justify the refusal.
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

    return result.to_dict()
