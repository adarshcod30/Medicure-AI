"""
Shared singletons, built once at startup.

The index is ~125 MB of sparse matrices and takes a couple of seconds to load.
Building it per request would be absurd; building it lazily on first request
would make the first user pay for everyone. It is constructed during the
lifespan startup so a failure surfaces at boot, when someone is watching, and
not in the middle of a scan.
"""

from __future__ import annotations

import logging

from packages.orchestrator import Orchestrator
from packages.pharmacology.price import CeilingPriceTable
from packages.reasoning.bedrock import BedrockClient
from packages.reasoning.explainer import Explainer
from packages.resolver.calibrate import Calibrator, load_or_default
from packages.resolver.index import BrandIndex

from .config import Settings

logger = logging.getLogger(__name__)


class AppState:
    """Everything the request handlers need."""

    def __init__(self) -> None:
        self.index: BrandIndex | None = None
        self.calibrator: Calibrator | None = None
        self.ceiling_table: CeilingPriceTable | None = None
        self.orchestrator: Orchestrator | None = None
        self.bedrock: BedrockClient | None = None
        self.startup_errors: list[str] = []

    @property
    def ready(self) -> bool:
        return self.orchestrator is not None

    def startup(self, settings: Settings) -> None:
        try:
            self.index = BrandIndex(settings.artifact_dir)
            logger.info("index loaded: %s", self.index.stats())
        except (FileNotFoundError, RuntimeError) as exc:
            # Fatal, and it must say so clearly. Without the index there is no
            # retrieval, and without retrieval this is just an LLM wrapper —
            # the exact thing the project exists not to be.
            self.startup_errors.append(f"index unavailable: {exc}")
            logger.error("index unavailable: %s", exc)
            return

        self.calibrator = load_or_default(settings.artifact_dir)
        if not self.calibrator.is_fitted:
            message = (
                "calibrator not fitted — confidence values are raw similarities, "
                "not probabilities. Run: python scripts/fit_calibrator.py"
            )
            self.startup_errors.append(message)
            logger.warning(message)

        self.ceiling_table = CeilingPriceTable(settings.data_dir)
        logger.info("ceiling prices: %s", self.ceiling_table.coverage)

        explainer = None
        if settings.enable_bedrock:
            try:
                self.bedrock = BedrockClient(
                    region=settings.aws_region,
                    model_id=settings.bedrock_model_id,
                    fast_model_id=settings.bedrock_fast_model_id,
                    max_tokens=settings.bedrock_max_tokens,
                    temperature=settings.bedrock_temperature,
                    guardrail_id=settings.bedrock_guardrail_id,
                    guardrail_version=settings.bedrock_guardrail_version,
                    access_key_id=settings.aws_access_key_id,
                    secret_access_key=settings.aws_secret_access_key,
                )
                explainer = Explainer(self.bedrock)
                logger.info("bedrock explainer enabled (%s)", settings.bedrock_fast_model_id)
            except Exception as exc:  # noqa: BLE001
                # Never fatal, and the catch is deliberately broad. Every
                # load-bearing capability is deterministic, so losing Bedrock
                # costs the prose and nothing else. A narrower catch let a
                # botocore MissingDependencyException escape and abort startup,
                # which took down retrieval, price checks and abstention over a
                # missing optional package.
                kind = type(exc).__name__
                self.startup_errors.append(f"bedrock unavailable ({kind}): {exc}")
                logger.warning("bedrock unavailable, continuing without explanations: %s", exc)

        self.orchestrator = Orchestrator(
            index=self.index,
            calibrator=self.calibrator,
            ceiling_table=self.ceiling_table,
            explainer=explainer,
        )


state = AppState()


def get_state() -> AppState:
    return state
