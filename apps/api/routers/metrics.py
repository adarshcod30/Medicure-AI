"""
Health and metrics.

`/v1/metrics` deliberately exposes the system's own limitations — index
coverage, calibration quality, and the NPPA provenance gap. A dashboard that
only reports successes cannot be used to decide whether to trust the thing.
"""

from __future__ import annotations

import json

from fastapi import APIRouter

from ..config import get_settings
from ..deps import get_state

router = APIRouter()


@router.get("/health")
def health() -> dict:
    """Readiness, plus exactly which capabilities are degraded and why."""
    state = get_state()
    settings = get_settings()

    capabilities = {
        "retrieval": state.index is not None,
        "calibrated_confidence": bool(state.calibrator and state.calibrator.is_fitted),
        "price_verification": state.ceiling_table is not None,
        "explanations": state.bedrock is not None,
    }

    return {
        "status": "ok" if state.ready else "not_ready",
        "ready": state.ready,
        "capabilities": capabilities,
        "degraded": state.startup_errors,
        "environment": settings.environment,
    }


@router.get("/metrics")
def metrics() -> dict:
    """Index coverage, calibration report and known data gaps."""
    state = get_state()
    settings = get_settings()

    payload: dict = {"ready": state.ready}

    if state.index:
        stats = state.index.stats()
        payload["index"] = {
            **stats,
            "brands_per_composition": round(
                stats["brands"] / max(stats["brand_signatures"], 1), 1
            ),
        }

    if state.ceiling_table:
        coverage = state.ceiling_table.coverage
        payload["price_data"] = {
            **coverage,
            "known_gap": (
                "nppa_notif and nppa_date are empty in the source dataset, so a "
                "ceiling price cannot be traced to the gazette order that set it. "
                "Closing this needs the NPPA gazette scrape."
            )
            if coverage["notification_coverage"] == 0.0
            else None,
        }

    report_path = settings.artifact_dir / "calibration_report.json"
    if report_path.exists():
        try:
            payload["calibration"] = json.loads(report_path.read_text())
        except (OSError, json.JSONDecodeError):
            payload["calibration"] = {"error": "calibration report unreadable"}
    else:
        payload["calibration"] = {
            "fitted": False,
            "note": "run scripts/fit_calibrator.py to produce calibrated confidence",
        }

    if state.calibrator and state.calibrator.is_fitted:
        payload["abstention_threshold"] = round(state.calibrator.threshold, 4)

    return payload
