"""
Application settings, read from the environment.

Every knob that differs between a laptop and a deployment lives here, and
nothing here has a secret as its default. `.env.example` documents the full set.

Note what is *not* configurable: the abstention threshold. It comes from the
fitted calibrator artifact, because it is a measured property of the model
rather than a preference. Exposing it as an environment variable would invite
raising it to make the system look more confident, which is precisely the
behaviour the calibration exists to prevent.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    # --- service ---
    app_name: str = "MediCure AI"
    environment: str = Field(default="development")
    api_port: int = 8000
    frontend_url: str = "http://localhost:5173"
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    # --- data ---
    data_dir: Path = REPO_ROOT / "data" / "processed"
    artifact_dir: Path = REPO_ROOT / "data" / "artifacts"

    # --- MongoDB ---
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_database: str = "medicure"

    # --- authentication ---
    jwt_secret: str = Field(default="dev-only-change-me")
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60 * 24 * 7
    google_client_id: str = ""

    # --- Amazon Bedrock ---
    aws_region: str = "us-east-1"

    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    """Optional explicit credentials, read from .env.

    Normally boto3's default chain (environment, ~/.aws/credentials, instance
    role) is the right thing and these stay empty. They exist because
    pydantic-settings reads .env into this object WITHOUT exporting to
    os.environ, so keys placed in .env are invisible to boto3 unless passed
    through explicitly. Silently falling back to a different identity than the
    one in .env is a confusing failure, so the wiring is made explicit."""

    bedrock_model_id: str = "us.amazon.nova-pro-v1:0"
    """Model for the explanation step.

    Defaults to Amazon Nova rather than Claude for a practical reason: Nova and
    Titan are **first-party AWS services**, while Anthropic, Meta, Mistral and
    Cohere models are delivered through **AWS Marketplace subscriptions**. The
    Marketplace path is what requires a valid payment instrument, so on an
    account without one the third-party models fail with
    INVALID_PAYMENT_INSTRUMENT while the first-party ones keep working.

    The choice matters far less here than it would in most systems. This model
    never produces a fact — it rephrases a fact sheet that retrieval and
    arithmetic have already filled in, and a Guardrail checks its output against
    that sheet. Model quality affects how the answer *reads*, not whether it is
    correct. Swapping families is a one-line change because the Converse API
    presents one request shape across every provider.

    Verify availability before deploying: `bash infra/aws/probe_models.sh`
    invokes each candidate for real, since listing a model proves nothing."""

    bedrock_fast_model_id: str = "us.amazon.nova-lite-v1:0"
    """Cheaper model for the explanation path, where the work is rephrasing
    rather than reasoning and the volume is high."""

    bedrock_embedding_model_id: str = "amazon.titan-embed-text-v2:0"

    bedrock_max_tokens: int = 2048
    """ALWAYS set explicitly on every call. Leaving maxTokens unset makes
    Bedrock reserve the model's maximum against your quota — tens of thousands
    of tokens for a response that needs hundreds — which produces
    ThrottlingException at request rates that look nowhere near the limit."""

    bedrock_temperature: float = 0.0
    """Zero by default. This system explains retrieved facts; there is nothing
    for sampling diversity to contribute, and it would only add variance
    between identical queries."""

    bedrock_guardrail_id: str = ""
    bedrock_guardrail_version: str = "DRAFT"

    enable_bedrock: bool = True
    """When false, the API runs fully without AWS: retrieval, price checks,
    alternatives and abstention all work, and only the natural-language
    explanation is omitted. Everything load-bearing is deterministic, so the
    system degrades rather than failing — and that is worth being able to
    demonstrate."""

    # --- image handling ---
    max_upload_bytes: int = 12 * 1024 * 1024
    store_uploads: bool = False
    """Off by default. The report claims no long-term storage of medical images;
    keeping the default at False makes the claim true rather than aspirational."""

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment.lower() in {"production", "prod"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
