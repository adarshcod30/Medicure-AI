"""
Vision transcription — the model reads characters, and nothing else.

This is the sharpest test of the project's governing rule, because a multimodal
model handed a photo of a medicine strip is *perfectly capable* of naming the
drug, and would usually be right. Letting it do so would be the single easiest
accuracy win available, and it is refused on purpose.

    The vision model transcribes. Retrieval identifies.

The reason is not purity. A model that names the drug is right most of the time
and confidently wrong the rest, with no way to tell the two apart — which is
exactly the failure the whole system exists to avoid. A model that returns
`"AUGMENTlN 625 DU0 / Amoxycillin 500mg"` feeds a resolver that will search
253,973 real products, score the match, and abstain if the evidence is thin.
The first path has no abstention available to it; the second does.

So the contract is narrow and enforced three ways:

1. **Prompt.** Transcribe verbatim; never infer, correct or name.
2. **Schema.** The return type is a list of text lines. There is no field for a
   drug name, a composition or a confidence in an identification.
3. **Downstream.** Output joins the OCR token bag and goes through the same
   resolver and the same calibration as Tesseract's. It gets no shortcut.

When it fires: only when the DIP quality gate says the image is degraded or
worse, since that is where Tesseract starts dropping characters and where the
extra cost is justified. A clean, well-resolved photo does not need it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import cv2
import numpy as np

from packages.reasoning.bedrock import BedrockClient, BedrockUnavailable

logger = logging.getLogger(__name__)

TRANSCRIBE_PROMPT = """You are a transcription tool for photographs of medicine packaging.

Your ONLY job is to read the characters that are visibly printed and write them out.

RULES

1. Transcribe EXACTLY what you can see, character for character, including text that
   looks misspelled, damaged or partial. If the pack shows "AUGMENTlN", write
   "AUGMENTlN" — do not correct it to "AUGMENTIN".
2. If part of a word is torn off, obscured or unreadable, transcribe only the part
   you can actually see. Write "UGMENTIN" if that is what is visible. Do not complete
   it.
3. Do NOT identify the medicine. Do NOT name the drug. Do NOT state what it treats,
   what its composition is, or what brand it is, unless those exact words are printed
   on the pack and you are reading them off it.
4. Do NOT guess at anything blurry. Omit it.
5. Include everything printed: brand text, composition, strengths, manufacturer,
   batch and expiry, price, warnings — in the order it appears.
6. Transcribe Devanagari or other non-Latin script as it appears, on its own line.

OUTPUT

One line of transcribed text per visual line on the pack. Nothing else — no preamble,
no commentary, no summary, no interpretation. If you can read nothing, output the
single word: NOTHING

You are a pair of eyes, not a pharmacist."""


@dataclass
class Transcription:
    """What the vision model read. Text only, by construction."""

    lines: list[str] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    available: bool = False
    reason: str | None = None
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0

    @property
    def text(self) -> str:
        return " ".join(self.lines)

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "lines": self.lines[:20],
            "token_count": len(self.tokens),
            "model": self.model,
            "reason": self.reason,
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "latency_ms": round(self.latency_ms, 1),
            },
        }


# Refusals and preambles a model emits instead of transcribing. Dropped so they
# never reach the token bag and get matched against a real medicine.
_NON_TRANSCRIPTION = re.compile(
    r"^\s*(nothing|i (cannot|can't|am unable)|unable to|sorry|the image shows|"
    r"this (image|appears|is a)|here (is|are)|transcription:|based on)",
    re.IGNORECASE,
)


def _encode(image: np.ndarray, *, max_dimension: int = 1600, quality: int = 90) -> bytes:
    """JPEG-encode for the API, bounded in size.

    Bounded because image tokens are charged by area and a 12 MP upload buys no
    additional legibility for text a model can already resolve at 1600 px.
    """
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest > max_dimension:
        scale = max_dimension / longest
        image = cv2.resize(
            image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA
        )

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("could not JPEG-encode image for vision transcription")
    return buffer.tobytes()


def _clean_lines(text: str) -> list[str]:
    """Split a model reply into transcribed lines, dropping non-transcription."""
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip().strip("-•*").strip()
        if not line or len(line) < 2:
            continue
        if _NON_TRANSCRIPTION.match(line):
            continue
        lines.append(line)
    return lines


class VisionTranscriber:
    """Reads characters off a packaging photo. Never identifies."""

    def __init__(self, client: BedrockClient, *, max_tokens: int = 800):
        self.client = client
        self.max_tokens = max_tokens
        self.last_error: str | None = None

    def transcribe(self, image: np.ndarray) -> Transcription:
        """Transcribe visible text. Returns an empty result rather than raising.

        Uses the *primary* model rather than the fast one: this is the only
        genuinely hard task given to a model in this system, and it is exactly
        where small-model errors turn into fabricated characters that then get
        matched against 253,973 real products.
        """
        from packages.perception.tesseract_engine import extract_tokens

        try:
            payload = _encode(image)
        except ValueError as exc:
            return Transcription(available=False, reason=str(exc))

        try:
            response = self.client.converse(
                system=TRANSCRIBE_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"image": {"format": "jpeg", "source": {"bytes": payload}}},
                            {"text": "Transcribe every character visible on this packaging."},
                        ],
                    }
                ],
                model_id=self.client.model_id,
                max_tokens=self.max_tokens,
                # No guardrail on this call. See BedrockClient.converse: the
                # contextual grounding filter needs a grounding source to score
                # against, and transcription has none — it is reading characters
                # off a photograph, not asserting facts. With the guardrail
                # attached it blocked every call and returned its own "could not
                # be verified" placeholder as the transcript.
                use_guardrail=False,
            )
        except BedrockUnavailable as exc:
            self.last_error = str(exc)[:200]
            logger.info("vision transcription unavailable: %s", exc)
            return Transcription(available=False, reason=self.last_error)

        # Belt and braces. Even without a guardrail attached to this call, a
        # blocked response must never be mistaken for a transcript: its
        # placeholder text is fluent English and tokenises into two dozen words
        # that then poison the retrieval query. Trusting response.text
        # unconditionally is what let that happen.
        if response.blocked_by_guardrail:
            self.last_error = "guardrail blocked the transcription response"
            logger.warning("vision transcription blocked by guardrail; ignoring output")
            return Transcription(available=False, reason=self.last_error)

        lines = _clean_lines(response.text)
        if not lines:
            return Transcription(
                available=True,
                reason="model read nothing legible",
                model=self.client.model_id,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                latency_ms=response.latency_ms,
            )

        return Transcription(
            lines=lines,
            tokens=extract_tokens(" ".join(lines)),
            available=True,
            model=self.client.model_id,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
        )


def merge_tokens(
    ocr_tokens: list[str], vision_tokens: list[str], *, cap: int = 60
) -> tuple[list[str], dict]:
    """Combine Tesseract and vision tokens, recording where each came from.

    Vision tokens go first. Not because the model is more trustworthy in
    general, but because it fires only when the quality gate has already judged
    Tesseract's input degraded — so in the cases where both exist, the vision
    read is the one with better odds. Tesseract's tokens still follow and still
    contribute; nothing is discarded on provenance alone.

    Attribution is returned so `/v1/scan` can show which stage produced the
    evidence behind an identification.
    """
    seen: set[str] = set()
    merged: list[str] = []
    origin: dict[str, str] = {}

    for token in vision_tokens:
        if token not in seen:
            seen.add(token)
            merged.append(token)
            origin[token] = "vision"

    for token in ocr_tokens:
        if token not in seen:
            seen.add(token)
            merged.append(token)
            origin[token] = "ocr"
        elif origin.get(token) == "vision":
            origin[token] = "both"

    merged = merged[:cap]
    return merged, {
        "vision_only": sum(1 for t in merged if origin[t] == "vision"),
        "ocr_only": sum(1 for t in merged if origin[t] == "ocr"),
        "agreed": sum(1 for t in merged if origin[t] == "both"),
    }
