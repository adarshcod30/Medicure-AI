"""
The model answering from its own knowledge, when retrieval has nothing.

Every other package here refuses to let a model originate a fact. This one is
the deliberate exception, and it exists because the alternative was measured
and found worse: a user who photographs Becosules gets "not in the catalogue"
and nothing else, when the model plainly knows it is a B-complex supplement.
Silence is not automatically the safer answer — it is only safer than an
answer the user cannot tell is unverified.

So the rule is not "never let the model speak". It is **never let the model
speak in the same voice as a cited fact**. Two mechanisms enforce that:

1. **Structural.** This never populates `ScanResult.facts`. It fills a separate
   `fallback` field carrying `verified: False`, the model id, and a disclaimer.
   Nothing merges the two, so a UI cannot accidentally render model output
   where a citation belongs.

2. **Guardrail, the half that still applies.** The contextual grounding filter
   cannot run here — it scores an answer against a source, and having no source
   is the entire situation. But the DENIED TOPICS policy has no such
   dependency, and it is the half that matters most: no diagnosis, no dosage
   advice, no safety assurances. `bedrock:ApplyGuardrail` checks the generated
   text directly, so the protection survives even though grounding cannot.

   Measured on the live guardrail:
       "Zincovit is a multivitamin supplement"     -> NONE
       "take two tablets every six hours"          -> BLOCKED (DosageAdvice)
       "you likely have a bacterial infection"     -> BLOCKED (MedicalDiagnosis)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .bedrock import BedrockClient, BedrockUnavailable

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "NOT VERIFIED. This came from the language model's own training, not from "
    "this system's medicine database. It carries no source and may be wrong or "
    "out of date. Treat it as a starting point and confirm with a pharmacist."
)

SYSTEM_PROMPT = """You are answering about a medicine that is NOT in this system's database.

You are the fallback, and the user has been told your answer is unverified. Be
useful, and be careful.

RULES

1. Say what you actually know. If you do not recognise the product, say so
   plainly rather than producing something plausible.
2. Never give a dose, a frequency, or a duration. Not "usually", not
   "typically", not "as directed on the pack".
3. Never diagnose, never interpret symptoms, never say a medicine is safe or
   unsafe for someone.
4. Never state a price. Prices are regional, change constantly, and this system
   verifies real ones against a government ceiling elsewhere.
5. Prefer the general and checkable over the specific and unverifiable: what
   class of product it is, what it broadly contains, what it is generally used
   for.
6. Three or four sentences. Plain words. Indian English.

If you genuinely do not know the product, the correct answer is one sentence
saying so."""


@dataclass
class UnverifiedAnswer:
    """A model-originated answer, labelled as such at every layer."""

    text: str | None = None
    available: bool = False
    reason: str = ""
    model: str = ""
    blocked_topics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "available": self.available,
            # Never omitted, never conditional. A consumer that forgets to
            # check this field still sees it is false.
            "verified": False,
            "source": "model_knowledge",
            "model": self.model,
            "disclaimer": DISCLAIMER,
            "reason": self.reason or None,
            "guardrail": {
                # Stated honestly rather than implied. Topic enforcement works
                # here; grounding cannot, because there is nothing to ground
                # against — that is what makes this the fallback.
                "denied_topics_enforced": True,
                "grounding_enforced": False,
                "blocked_topics": self.blocked_topics,
            },
        }


class FallbackAnswerer:
    """Asks the model, then screens what it said against the denied topics."""

    def __init__(self, client: BedrockClient, *, max_tokens: int = 400) -> None:
        self.client = client
        self.max_tokens = max_tokens

    def answer(self, question: str, *, subject: str = "") -> UnverifiedAnswer:
        prompt = (
            f"The user asked about: {subject}\n\n{question}"
            if subject
            else question
        )
        try:
            response = self.client.converse(
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                model_id=self.client.model_id,
                max_tokens=self.max_tokens,
                # No contextual grounding: there is no source. Denied topics
                # are enforced below instead, on the generated text.
                use_guardrail=False,
            )
        except BedrockUnavailable as exc:
            logger.info("fallback unavailable: %s", exc)
            return UnverifiedAnswer(reason=str(exc)[:200])

        text = (response.text or "").strip()
        if not text:
            return UnverifiedAnswer(reason="model returned nothing", model=self.client.model_id)

        blocked = self._denied_topics(text)
        if blocked:
            logger.info("fallback answer blocked by guardrail topics: %s", blocked)
            return UnverifiedAnswer(
                reason=(
                    "The model's answer strayed into advice this system will not give "
                    f"({', '.join(blocked)}), so it was withheld. Please ask a pharmacist."
                ),
                model=self.client.model_id,
                blocked_topics=blocked,
            )

        return UnverifiedAnswer(text=text, available=True, model=self.client.model_id)

    def _denied_topics(self, text: str) -> list[str]:
        """Topic names the guardrail objects to, or an empty list.

        A failure here returns no topics rather than raising: the answer is
        already labelled unverified, and losing the fallback because the
        screening call timed out helps nobody. The structural separation is the
        load-bearing protection; this is defence in depth on top of it.
        """
        if not self.client.guardrail_id:
            return []
        try:
            result = self.client.apply_guardrail_to_output(text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("guardrail screening failed, allowing labelled answer: %s", exc)
            return []
        return result
