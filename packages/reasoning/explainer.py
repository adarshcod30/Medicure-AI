"""
The explanation layer — the only place a language model speaks.

By the time this runs, the identification, the price arithmetic and the
alternatives are already decided and already carry their sources. The model's
entire job is to render those facts as a few plain sentences a person without
medical training can act on.

Three mechanisms keep it there, in decreasing order of how much they can be
trusted:

1. **Structural.** The model is handed a rendered fact sheet and nothing else —
   no retrieval tools, no memory of the query. Its output goes into
   `explanation.text` and no other field. A hallucinated price cannot reach
   `price_check.overcharge_percent`, because the model never writes there.

2. **Guardrail.** Bedrock's contextual grounding check scores the answer against
   the fact sheet and blocks it below threshold. Mechanical, and independent of
   whether the model chose to comply.

3. **Prompt.** Instructions to invent nothing. Weakest of the three, and the
   only one the replaced system relied on — which is why its prompt could say
   "add approximate Indian price inside brackets" and get exactly that.

Note the inversion: the prompt here forbids adding facts, where the prompt it
replaces demanded them ("ALWAYS give at least 2-3 cheaper alternatives...
NEVER leave the list empty"). Same technology, opposite instruction, opposite
failure mode.
"""

from __future__ import annotations

import logging

from packages.reasoning.bedrock import BedrockClient, BedrockUnavailable

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are MediCure, explaining a medicine to someone in India with no medical training.

You will be given a FACT SHEET that has already been checked against government and
pharmaceutical databases. Your only job is to put those facts into plain words.

ABSOLUTE RULES

1. Use ONLY what is in the fact sheet. If it is not there, you do not know it.
2. Never state a price, a saving, a brand name, a side effect, an interaction or a
   dose that is not written in the fact sheet. Not "approximately", not "usually",
   not "typically". If the fact sheet has no price, do not mention price.
3. If the fact sheet says no cheaper alternative exists, say that plainly. Do not
   suggest one. Do not say "ask your pharmacist about generics" as a substitute for
   naming one — an empty answer is the correct answer.
4. If the fact sheet says confidence is low or the medicine is uncertain, LEAD with
   that. Do not bury it after a confident-sounding description.
5. Never give a diagnosis, never recommend starting, stopping or changing a dose,
   and never say a medicine is safe for someone.

STYLE

- Short sentences. Everyday words: "painkiller" not "analgesic", "fever medicine"
  not "antipyretic", "swelling" not "inflammation".
- Indian English. Rupees as "Rs".
- 4 to 6 sentences. No headings, no bullet points, no markdown.
- Do not open with "This medicine is". Just say what it is.

If the fact sheet is thin, write less. A short honest answer is correct; a long
one padded with general medical knowledge is not."""


def render_fact_sheet(result) -> str:
    """Render a `ScanResult` into the only context the model is given.

    Also serves as the grounding source for the contextual grounding check, so
    it must contain every fact the explanation is permitted to state — and
    nothing beyond, since anything absent here is by definition ungrounded.
    """
    identification = result.identification
    lines: list[str] = ["=== FACT SHEET ==="]

    lines.append(f"Confidence in identification: {identification.probability:.0%}")
    lines.append(f"Status: {identification.status}")
    if identification.reason:
        lines.append(f"Note: {identification.reason}")

    if identification.composition:
        lines.append(f"Active ingredients: {identification.composition}")
    if identification.closest_brand:
        lines.append(f"Closest matching product: {identification.closest_brand}")
    if identification.brands_sharing_composition:
        lines.append(
            f"{identification.brands_sharing_composition} products in the Indian market "
            "share this exact composition."
        )

    product = result.reference_product
    if product:
        lines.append(f"Manufacturer: {product.manufacturer or 'not recorded'}")
        lines.append(f"Dosage form: {product.dosage_form or 'not recorded'}")
        if product.price is not None:
            lines.append(
                f"Listed price: Rs {product.price:.2f} for {product.pack_label or 'one pack'}"
            )

    price = result.price_check
    if price:
        lines.append("")
        lines.append(f"PRICE CHECK ({price.status}): {price.message}")
        for step in price.workings:
            lines.append(f"  working: {step}")

    alternatives = result.alternatives
    if alternatives:
        lines.append("")
        lines.append(f"CHEAPER ALTERNATIVES: {alternatives.message}")
        if alternatives.alternatives:
            for alternative in alternatives.alternatives[:4]:
                if alternative.implausible:
                    continue
                label = (
                    "Jan Aushadhi" if alternative.kind == "jan_aushadhi" else "another brand"
                )
                lines.append(
                    f"  - {alternative.name} ({label}), Rs {alternative.price_per_unit:.2f} "
                    f"per unit, about {alternative.saving_percent:.0f}% cheaper"
                )
        else:
            lines.append(
                "  - NONE. There is no cheaper equivalent to name. Say so; do not suggest one."
            )

    quality = result.image_quality
    if quality and quality.get("verdict") not in (None, "good"):
        lines.append("")
        lines.append(f"IMAGE QUALITY: {quality['verdict']} — {'; '.join(quality.get('reasons', []))}")

    lines.append("=== END OF FACT SHEET ===")
    return "\n".join(lines)


class Explainer:
    """Turns a `ScanResult` into plain prose. Never into new facts."""

    def __init__(self, client: BedrockClient, *, fast: bool = True):
        self.client = client
        self.fast = fast
        """Explanation is rephrasing, not reasoning, so the cheaper model is
        the right default. Roughly a tenth of the cost for output that is
        indistinguishable on this task."""

    def explain(self, result) -> dict:
        fact_sheet = render_fact_sheet(result)

        question = (
            "Explain this medicine in plain words for someone with no medical background."
        )
        instruction = f"{fact_sheet}\n\n{question}"

        # The contextual grounding filter needs THREE things and names them:
        # a grounding source, a query, and the content to guard. The source is
        # attached to the system block by BedrockClient.converse; the query is
        # this block, and it MUST carry the "query" qualifier. Sending an
        # unqualified guardContent block instead fails the whole call with
        # "The provided request does not contain the query", which reads like a
        # malformed request rather than a missing qualifier.
        #
        # Only the question goes here, not the fact sheet: the sheet already
        # reaches the model through the grounding_source block, and repeating
        # it inside the query would ask the guardrail to score the answer
        # against text that is also part of the question.
        #
        # guardContent is still only valid when a guardrail is attached, so
        # without one the whole prompt goes as ordinary text.
        content = (
            [{"guardContent": {"text": {"text": question, "qualifiers": ["query"]}}}]
            if self.client.guardrail_id
            else [{"text": instruction}]
        )

        try:
            response = self.client.converse(
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
                grounding_source=fact_sheet,
                model_id=self.client.fast_model_id if self.fast else self.client.model_id,
                max_tokens=600,
            )
        except BedrockUnavailable as exc:
            logger.info("explanation skipped: %s", exc)
            return {
                "text": None,
                "available": False,
                "reason": str(exc)[:200],
                "note": (
                    "The explanation is unavailable, but every fact above was "
                    "retrieved and computed without it."
                ),
            }

        if response.blocked_by_guardrail:
            return {
                "text": None,
                "available": False,
                "reason": "blocked by guardrail",
                "note": (
                    "The generated explanation was not sufficiently grounded in the "
                    "retrieved facts and was withheld."
                ),
                "usage": response.to_dict(),
            }

        return {
            "text": response.text,
            "available": True,
            "grounded_against": "fact_sheet",
            "guardrail_enforced": bool(self.client.guardrail_id),
            "grounding_note": None
            if self.client.guardrail_id
            else (
                "No Bedrock guardrail is configured, so groundedness rests on the "
                "prompt contract rather than a mechanical check. Set "
                "BEDROCK_GUARDRAIL_ID to enforce it."
            ),
            "model": self.client.fast_model_id if self.fast else self.client.model_id,
            "usage": response.to_dict(),
        }
