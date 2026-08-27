"""
Amazon Bedrock client — the LLM layer, deliberately kept small.

Uses the **Converse API** rather than InvokeModel: one request shape across
providers, instead of a different JSON body per model family where using the
wrong one produces "Malformed input request".

That choice paid off. Switching the whole system from Claude to Amazon Nova
touched one default parameter and two lines of config — no request-body
rewrite, no per-provider branching. Under InvokeModel the same switch would
have meant rewriting the payload for a different provider's schema.

Two settings are not negotiable here:

**maxTokens is always set explicitly.** Leaving it unset makes Bedrock reserve
the model's maximum against your account quota — tens of thousands of tokens
for a reply that needs a few hundred. The result is ThrottlingException at
request rates nowhere near the documented limit, which is then extremely hard
to diagnose because the numbers look fine.

**temperature is 0.** This system explains facts that have already been
retrieved and computed. Sampling diversity has nothing to contribute and would
only make two identical scans produce two different explanations.

Retries use adaptive mode and distinguish retryable failures (throttling,
timeouts, 5xx) from permanent ones (validation, access denied, not found).
Retrying a ValidationException just fails four more times.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

try:
    import boto3
    from botocore.config import Config

    BOTO3_AVAILABLE = True
except ImportError:  # pragma: no cover
    boto3 = None
    Config = None
    BOTO3_AVAILABLE = False


RETRYABLE = {
    "ThrottlingException",
    "ModelTimeoutException",
    "ServiceUnavailableException",
    "InternalServerException",
    "ModelNotReadyException",
}

PERMANENT = {
    "ValidationException",
    "AccessDeniedException",
    "ResourceNotFoundException",
    "UnrecognizedClientException",
}


class BedrockUnavailable(RuntimeError):
    """Bedrock cannot be reached or is not configured.

    Raised rather than swallowed so callers decide. The orchestrator's answer
    is already complete without it, so the right handling is to omit the prose
    and say so — not to fail the request.
    """


@dataclass
class LlmResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = ""
    guardrail_action: str | None = None
    latency_ms: float = 0.0

    @property
    def blocked_by_guardrail(self) -> bool:
        """Did the guardrail withhold this answer?

        Case-folded deliberately. The Converse API returns `stopReason` as
        lowercase `guardrail_intervened`, while the InvokeModel-era docs and
        the console both show `GUARDRAIL_INTERVENED`. Comparing against the
        upper-case spelling silently returned False for every intervention, so
        the explainer passed the guardrail's own "withheld" placeholder through
        as if it were a real explanation — reporting available=True for an
        answer that had just been blocked. The bug was invisible until a
        guardrail actually existed to intervene.
        """
        return (self.guardrail_action or "").upper() == "GUARDRAIL_INTERVENED"

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "stop_reason": self.stop_reason,
            "guardrail_action": self.guardrail_action,
            "latency_ms": round(self.latency_ms, 1),
        }


class BedrockClient:
    """Thin wrapper over `bedrock-runtime`."""

    def __init__(
        self,
        *,
        region: str = "us-east-1",
        model_id: str = "us.amazon.nova-pro-v1:0",
        fast_model_id: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.0,
        guardrail_id: str = "",
        guardrail_version: str = "DRAFT",
        access_key_id: str = "",
        secret_access_key: str = "",
    ):
        if not BOTO3_AVAILABLE:
            raise BedrockUnavailable("boto3 is not installed")

        self.region = region
        self.model_id = model_id
        self.fast_model_id = fast_model_id or model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.guardrail_id = guardrail_id
        self.guardrail_version = guardrail_version

        # Invocation outcome, so health reporting can distinguish "a client was
        # constructed" from "the model can actually be called". They are not
        # the same: an account with no valid payment instrument builds a client
        # happily and then fails every Converse with AccessDeniedException.
        # Reporting the first as a working capability is the kind of green
        # dashboard that hides an outage.
        self.last_error: str | None = None
        self.last_success: bool | None = None

        # Every construction failure becomes BedrockUnavailable, deliberately.
        #
        # boto3 raises a wide and not-fully-enumerable set of exceptions here:
        # NoCredentialsError, ProfileNotFound, NoRegionError, and — encountered
        # in practice — MissingDependencyException, because the `aws login`
        # credential provider needs botocore[crt] and says so only at client
        # construction. Catching a curated list let that one escape and take
        # the whole service down at startup, which is precisely the opposite of
        # the design: identification, price checks and abstention are all
        # deterministic and must keep working when the LLM layer does not.
        # Explicit credentials only when .env supplied them. pydantic-settings
        # reads .env into the Settings object without exporting to os.environ,
        # so keys placed there are invisible to boto3's default chain — the
        # client would silently authenticate as whatever `aws configure` last
        # left behind, which is a confusing way to fail.
        credentials = {}
        if access_key_id and secret_access_key:
            credentials = {
                "aws_access_key_id": access_key_id,
                "aws_secret_access_key": secret_access_key,
            }
            self.credential_source = "env"
        else:
            self.credential_source = "default_chain"

        try:
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=region,
                config=Config(
                    retries={"max_attempts": 4, "mode": "adaptive"},
                    connect_timeout=5,
                    read_timeout=60,
                ),
                **credentials,
            )
        except Exception as exc:  # noqa: BLE001 - see above
            raise BedrockUnavailable(
                f"could not create bedrock-runtime client: {type(exc).__name__}: {exc}"
            ) from exc

    # --- guardrail config -------------------------------------------------

    def _guardrail_config(self, *, streaming: bool = False) -> dict | None:
        if not self.guardrail_id:
            return None
        config = {
            "guardrailIdentifier": self.guardrail_id,
            "guardrailVersion": self.guardrail_version,
            # Trace stays disabled: enabling it returns the original text that
            # triggered a filter, which for a medical app can include the
            # user's own health details, and that would then be logged.
            "trace": "disabled",
        }
        if streaming:
            # Synchronous: chunks are evaluated before reaching the user.
            # Async streaming delivers content before the guardrail sees it and
            # does not mask PII at all.
            config["streamProcessingMode"] = "sync"
        return config

    # --- invocation -------------------------------------------------------

    def converse(
        self,
        *,
        system: str,
        messages: list[dict],
        grounding_source: str | None = None,
        model_id: str | None = None,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
    ) -> LlmResponse:
        """One Converse call.

        `grounding_source` is wrapped in a `guardContent` block qualified as
        `grounding_source`, which is what enables the contextual grounding
        check: the guardrail scores the model's answer against this text and
        blocks it if unsupported. That check is the mechanical enforcement of
        "the LLM only explains what was retrieved" — without it the instruction
        is a request the model may decline.
        """
        import time

        request: dict[str, Any] = {
            "modelId": model_id or self.model_id,
            "messages": messages,
            "system": [{"text": system}],
            # Always explicit. See the module docstring.
            "inferenceConfig": {
                "maxTokens": max_tokens or self.max_tokens,
                "temperature": self.temperature,
            },
        }

        # guardContent blocks are ONLY valid when a guardrail is attached.
        # Bedrock rejects them otherwise with "The guardrail can't assess the
        # content in the guardContent field", which is a confusing way to say
        # "you did not configure a guardrail". Without one the grounding source
        # still goes to the model — as ordinary system text — so the prompt
        # contract holds; what is lost is the *mechanical* grounding check.
        if grounding_source:
            if self.guardrail_id:
                request["system"].append(
                    {
                        "guardContent": {
                            "text": {"text": grounding_source, "qualifiers": ["grounding_source"]}
                        }
                    }
                )
            else:
                request["system"].append({"text": grounding_source})

        guardrail = self._guardrail_config()
        if guardrail:
            request["guardrailConfig"] = guardrail

        if tools:
            request["toolConfig"] = {"toolSpec": tools} if isinstance(tools, dict) else {
                "tools": tools
            }

        started = time.perf_counter()
        try:
            response = self._client.converse(**request)
        except Exception as exc:  # noqa: BLE001
            code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
            self.last_success = False
            self.last_error = f"{code or type(exc).__name__}: {str(exc)[:240]}"

            if "INVALID_PAYMENT_INSTRUMENT" in str(exc):
                # Worth naming explicitly: this is an AWS billing state, not a
                # Bedrock permission or a model-access setting, and hunting for
                # it in IAM policies wastes a lot of time.
                self.last_error = (
                    "AWS account has no valid payment instrument. This is a billing "
                    "setting, not a Bedrock permission — add a payment method to the "
                    "account, then retry."
                )

            if code in PERMANENT:
                raise BedrockUnavailable(self.last_error) from exc
            logger.warning("bedrock converse failed (%s): %s", code or type(exc).__name__, exc)
            raise BedrockUnavailable(self.last_error) from exc

        self.last_success = True
        self.last_error = None

        latency = (time.perf_counter() - started) * 1000

        content = response.get("output", {}).get("message", {}).get("content", [])
        text = "".join(block.get("text", "") for block in content)
        usage = response.get("usage", {})

        return LlmResponse(
            text=text.strip(),
            input_tokens=int(usage.get("inputTokens", 0)),
            output_tokens=int(usage.get("outputTokens", 0)),
            stop_reason=response.get("stopReason", ""),
            # Case-folded on the way in as well, so the two spellings AWS uses
            # for this value cannot diverge from `blocked_by_guardrail`.
            guardrail_action=response.get("stopReason")
            if (response.get("stopReason") or "").upper() == "GUARDRAIL_INTERVENED"
            else None,
            latency_ms=latency,
        )

    def apply_guardrail(self, text: str, *, source: str = "OUTPUT") -> dict:
        """Evaluate text against the guardrail without invoking a model.

        Used to score groundedness of an already-generated explanation, and by
        `eval/bench_groundedness.py` to measure it in bulk.
        """
        if not self.guardrail_id:
            return {"action": "NONE", "reason": "no guardrail configured"}

        response = self._client.apply_guardrail(
            guardrailIdentifier=self.guardrail_id,
            guardrailVersion=self.guardrail_version,
            source=source,
            content=[{"text": {"text": text}}],
        )
        return {
            "action": response.get("action", "NONE"),
            "outputs": response.get("outputs", []),
            "assessments": response.get("assessments", []),
        }

    def health(self) -> dict:
        """Capability report for `/v1/health`.

        Reports the outcome of the last real invocation rather than probing,
        because a probe on every health check costs tokens and a probe at
        startup goes stale immediately.
        """
        return {
            "client_constructed": True,
            "region": self.region,
            "model": self.model_id,
            "fast_model": self.fast_model_id,
            "guardrail": bool(self.guardrail_id),
            "credential_source": self.credential_source,
            "last_invocation_succeeded": self.last_success,
            "last_error": self.last_error,
        }


def parse_json_response(text: str) -> dict | None:
    """Extract a JSON object from a model reply.

    Models wrap JSON in prose or fences despite instructions. This recovers the
    outermost braced span rather than trusting the whole reply to parse — and
    returns None rather than a partially-repaired object, because a silently
    mangled structure is worse than an absent one.
    """
    if not text:
        return None

    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("```")[1] if "```" in candidate[3:] else candidate[3:]
        candidate = candidate.removeprefix("json").strip()

    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end <= start:
        return None

    try:
        return json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return None
