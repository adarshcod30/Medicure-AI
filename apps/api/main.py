"""
MediCure AI — the single backend service.

Consolidated from what was previously a Node/Express gateway in front of a
Python ML service. One language, one deployment, and no serialisation boundary
between the code that retrieves facts and the code that serves them.

Endpoints:

    GET  /v1/health     readiness, including which capabilities degraded
    POST /v1/scan       image  -> grounded, cited answer
    POST /v1/search     text   -> the same contract
    GET  /v1/metrics    index coverage, calibration report, data-quality gaps

Every response carries provenance. There is no field in the schema for a fact
that was not retrieved or computed.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .deps import state
from .routers import auth, cabinet, history, interactions, lasa, metrics, scan, search

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
)
logger = logging.getLogger("medicure")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("starting %s (%s)", settings.app_name, settings.environment)

    state.startup(settings)
    await state.connect_store(settings)

    if state.ready:
        logger.info("ready")
    else:
        # Serve anyway so /v1/health can explain what is wrong. A process that
        # exits on a missing artifact tells an operator nothing beyond "it
        # crashed"; one that reports "run scripts/build_index.py" is actionable.
        logger.error("NOT READY: %s", "; ".join(state.startup_errors))

    for problem in state.startup_errors:
        logger.warning("degraded: %s", problem)

    yield
    if state.store is not None:
        state.store.close()
    logger.info("shutting down")


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="2.0.0",
    description=(
        "Grounded medicine safety and affordability engine. "
        "DIP restores the image, retrieval identifies the drug, "
        "and the language model only explains what was retrieved."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

app.include_router(scan.router, prefix="/v1", tags=["scan"])
app.include_router(search.router, prefix="/v1", tags=["search"])
app.include_router(metrics.router, prefix="/v1", tags=["metrics"])
app.include_router(auth.router, prefix="/v1/auth", tags=["auth"])
app.include_router(history.router, prefix="/v1", tags=["history"])
app.include_router(cabinet.router, prefix="/v1", tags=["cabinet"])
app.include_router(interactions.router, prefix="/v1", tags=["interactions"])
app.include_router(lasa.router, prefix="/v1", tags=["lasa"])


@app.get("/")
def root() -> dict:
    return {
        "service": settings.app_name,
        "version": "2.0.0",
        "ready": state.ready,
        "docs": "/docs",
    }
