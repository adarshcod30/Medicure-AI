"""
Follow-up questions about a medicine that has already been identified.

The feature the replaced system had, rebuilt so its answers can be checked.
Before, a chatbot answered everything from the model's own memory; a question
about side effects and a question about price were served the same way, and
neither carried a source.

Here a question takes one of two paths, and the user is always told which:

  GROUNDED   the answer comes from the fact sheet — composition, price
             working, alternatives, uses, side effects, interactions — and the
             contextual grounding guardrail scores it against that sheet
             before it is returned. Blocked answers are withheld.

  UNVERIFIED retrieval carried nothing relevant, so the model answers from its
             own training, labelled verified=False and screened against the
             denied-topic policy.

The grounding source is the whole fact sheet, not passages picked by a
retriever. That was tested rather than assumed: narrowing the source to just
the clinical section scored WORSE on the grounding filter (0.07 against 0.21),
because the filter scores the answer against whatever it is given and a
smaller source is not automatically a better-matched one.

What actually mattered was subtler and worth remembering. "What are the side
effects?" was being blocked at 0.21 while the model's answer restated the sheet
almost verbatim — because it appended "however, not everyone will experience
these", a sentence this module's own prompt had asked for and which appears
nowhere in the source. One self-authored hedge is enough to fail the whole
answer. Moving that caveat INTO the fact sheet took the same question from 0.21
(blocked) to 0.99 (passed).

The rule that falls out: anything the answer is allowed to say must exist in
the source. Prompts may shape tone; they must not introduce content.

Conversation history is included so follow-ups resolve ("what about its side
effects?" after "what is this for?"), but history is NEVER a source. Only the
fact sheet grounds an answer; earlier turns just disambiguate the question.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .bedrock import BedrockClient, BedrockUnavailable
from .explainer import render_fact_sheet

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 6
"""Enough for a follow-up to resolve its pronouns, bounded so a long session
cannot push the fact sheet out of the model's attention — the sheet is the
only thing that grounds the answer and it must stay prominent."""

SYSTEM_PROMPT = """You answer follow-up questions about ONE medicine, using ONLY the fact sheet provided.

The fact sheet has already been checked against government and pharmaceutical
databases. Everything you say must be traceable to it.

ABSOLUTE RULES

1. Use ONLY the fact sheet. If it does not contain the answer, say "the fact
   sheet does not cover that" and stop. Do not fill the gap from memory.
2. Never give a dose, a frequency or a duration, even if the user asks
   directly, and even if the fact sheet mentions strengths.
3. Never diagnose, never interpret symptoms, never say the medicine is safe or
   unsafe for a particular person or condition.
4. Quote figures exactly as the sheet gives them. Do not round prices, do not
   convert units, do not estimate.
5. Do not add caveats, hedges or reassurances of your own. If the fact sheet
   carries a caveat, repeat it; if it does not, do not invent one. A sentence
   you added yourself is unsupported by definition, and the grounding check
   will withhold the entire answer because of it.
6. Two to four sentences. Plain words. Indian English. Rupees as "Rs".

If the user asks something the sheet cannot answer, saying so plainly IS the
correct answer."""


@dataclass
class ChatTurn:
    """One answer, with its provenance made explicit."""

    text: str | None = None
    available: bool = False
    grounded: bool = True
    reason: str = ""
    model: str = ""
    blocked_topics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "available": self.available,
            # The field a UI must branch on. True means every claim traces to
            # the fact sheet; False means the model answered from training.
            "grounded": self.grounded,
            "verified": self.grounded,
            "source": "fact_sheet" if self.grounded else "model_knowledge",
            "model": self.model,
            "reason": self.reason or None,
            "disclaimer": None if self.grounded else (
                "NOT VERIFIED. This came from the language model's own training, "
                "not from this system's databases. Confirm with a pharmacist."
            ),
            "guardrail": {
                "grounding_enforced": self.grounded,
                "denied_topics_enforced": True,
                "blocked_topics": self.blocked_topics,
            },
        }


class ChatAnswerer:
    """Grounded Q&A over a single ScanResult, with a labelled fallback."""

    def __init__(
        self,
        client: BedrockClient,
        *,
        fallback_answerer=None,
        max_tokens: int = 500,
    ) -> None:
        self.client = client
        self.fallback_answerer = fallback_answerer
        self.max_tokens = max_tokens

    def answer(
        self, question: str, result, history: list[dict] | None = None
    ) -> ChatTurn:
        fact_sheet = render_fact_sheet(result)
        messages = self._messages(question, history or [])

        try:
            response = self.client.converse(
                system=SYSTEM_PROMPT,
                messages=messages,
                grounding_source=fact_sheet,
                model_id=self.client.fast_model_id,
                max_tokens=self.max_tokens,
            )
        except BedrockUnavailable as exc:
            logger.info("chat unavailable: %s", exc)
            return ChatTurn(reason=str(exc)[:200], grounded=True)

        if not response.blocked_by_guardrail and (response.text or "").strip():
            return ChatTurn(
                text=response.text.strip(),
                available=True,
                grounded=True,
                model=self.client.fast_model_id,
            )

        # The grounded attempt was withheld — either the guardrail judged it
        # unsupported by the sheet, or the model produced nothing. Both mean
        # the same thing to the user: the databases do not answer this. Offer
        # the model's own knowledge, clearly marked.
        logger.info("chat answer not grounded; falling back (blocked=%s)",
                    response.blocked_by_guardrail)
        if self.fallback_answerer is None:
            return ChatTurn(
                reason=(
                    "That is not covered by the retrieved records for this medicine, "
                    "and no unverified fallback is configured on this deployment."
                ),
                grounded=True,
            )

        unverified = self.fallback_answerer.answer(
            question, subject=_subject(result)
        )
        return ChatTurn(
            text=unverified.text,
            available=unverified.available,
            grounded=False,
            reason=unverified.reason,
            model=unverified.model,
            blocked_topics=unverified.blocked_topics,
        )

    def _messages(self, question: str, history: list[dict]) -> list[dict]:
        """Prior turns for pronoun resolution, then the question as the query.

        The question carries the "query" qualifier the contextual grounding
        filter requires. History does not: it is context for understanding what
        was asked, never a source for answering it, and qualifying it would
        invite the guardrail to treat an earlier model turn as ground truth.
        """
        messages: list[dict] = []
        for turn in history[-MAX_HISTORY_TURNS:]:
            role = turn.get("role")
            text = (turn.get("text") or "").strip()
            if role in {"user", "assistant"} and text:
                messages.append({"role": role, "content": [{"text": text}]})

        if self.client.guardrail_id:
            content = [{"guardContent": {"text": {"text": question, "qualifiers": ["query"]}}}]
        else:
            content = [{"text": question}]
        messages.append({"role": "user", "content": content})
        return messages


def _subject(result) -> str:
    identification = getattr(result, "identification", None)
    if identification is None:
        return ""
    return identification.closest_brand or identification.composition or ""
