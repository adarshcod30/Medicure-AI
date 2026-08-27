"""
Tests for vision transcription.

The valuable ones assert what the model must NOT do. A multimodal model handed
a photo of a strip can name the drug and would usually be right — that is the
easiest accuracy win available here, and refusing it is the entire point. These
tests pin the refusal so it cannot quietly erode.
"""

from __future__ import annotations

import numpy as np
import pytest

from packages.perception.vision_transcribe import (
    TRANSCRIBE_PROMPT,
    Transcription,
    VisionTranscriber,
    _clean_lines,
    _encode,
    merge_tokens,
)


class FakeClient:
    """A BedrockClient stand-in that returns a scripted reply."""

    def __init__(self, text="", raise_with=None):
        self.model_id = "us.amazon.nova-pro-v1:0"
        self.fast_model_id = "us.amazon.nova-lite-v1:0"
        self.guardrail_id = ""
        self._text = text
        self._raise = raise_with
        self.last_request = None

    def converse(self, **kwargs):
        from packages.reasoning.bedrock import BedrockUnavailable, LlmResponse

        self.last_request = kwargs
        if self._raise:
            raise BedrockUnavailable(self._raise)
        return LlmResponse(text=self._text, input_tokens=100, output_tokens=20, latency_ms=500.0)


def strip_image() -> np.ndarray:
    from tests import synthetic

    return synthetic.make_strip()


# --- the contract ---------------------------------------------------------


def test_prompt_forbids_identification():
    """The transcription contract, asserted on the prompt itself.

    Whitespace is normalised first: the prompt is hard-wrapped for readability,
    so a phrase can span a line break. The contract is about wording, not
    formatting, and a test that breaks when a line is re-wrapped is testing the
    wrong thing.
    """
    lowered = " ".join(TRANSCRIBE_PROMPT.lower().split())
    assert "do not identify" in lowered
    assert "do not name the drug" in lowered
    assert "do not correct" in lowered
    # And it must forbid completing partial words, not just correcting them.
    assert "do not complete it" in lowered


def test_result_has_no_field_for_an_identification():
    """Structural enforcement: there is nowhere to put a drug name.

    Even a model that ignored every instruction could not smuggle an
    identification through this type — it returns lines of text and token
    counts, and nothing else.
    """
    fields = set(Transcription.__dataclass_fields__)
    for forbidden in ("composition", "drug", "brand", "medicine", "identification",
                      "diagnosis", "confidence_in_drug"):
        assert forbidden not in fields


def test_transcription_preserves_ocr_style_errors():
    """A misread character must survive, not be silently corrected.

    Observed on a real photo: the model returned "NostrosiI" with a capital I,
    and "stro-resistant" as the visible tail of "gastro-resistant". Both are
    correct behaviour — the resolver's character n-grams handle damaged input,
    but only if the damage reaches them rather than being tidied away into a
    confident wrong word.
    """
    client = FakeClient("AUGMENTlN 625 DU0\nAmoxycillin 500mg")
    result = VisionTranscriber(client).transcribe(strip_image())

    assert result.available
    assert "augmentln" in result.tokens  # the l-for-I misread is preserved
    assert "augmentin" not in result.tokens


# --- refusals and noise ---------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        "I cannot read this image.",
        "Sorry, the image is too blurry.",
        "This image shows a medicine strip.",
        "Here is the transcription:",
        "NOTHING",
    ],
)
def test_model_preamble_and_refusals_are_dropped(reply):
    """A refusal must not become a token the resolver matches on.

    "This image shows a medicine strip" contains `medicine` and `strip`, which
    would be searched against 253,973 real products as though the pack had said
    them.
    """
    assert _clean_lines(reply) == []


def test_genuine_transcription_survives_cleaning():
    lines = _clean_lines("Combiflam\nIbuprofen and Paracetamol Tablets\nSANOFI")
    assert len(lines) == 3
    assert "Combiflam" in lines


def test_empty_reply_is_reported_not_invented():
    result = VisionTranscriber(FakeClient("NOTHING")).transcribe(strip_image())
    assert result.available
    assert result.tokens == []
    assert result.reason


def test_bedrock_failure_degrades_quietly():
    """A vision failure must not take the scan down; OCR still ran."""
    result = VisionTranscriber(FakeClient(raise_with="throttled")).transcribe(strip_image())
    assert result.available is False
    assert result.tokens == []
    assert "throttled" in (result.reason or "")


# --- encoding and merging -------------------------------------------------


def test_encoding_bounds_image_size():
    """Image tokens are charged by area; a 12 MP upload buys no legibility."""
    big = np.full((4000, 3000, 3), 200, dtype=np.uint8)
    payload = _encode(big, max_dimension=1600)
    assert len(payload) > 0

    import cv2

    decoded = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    assert max(decoded.shape[:2]) <= 1600


def test_merge_prefers_vision_but_keeps_everything():
    merged, attribution = merge_tokens(
        ocr_tokens=["store", "tablets", "combiflam"],
        vision_tokens=["combiflam", "ibuprofen", "paracetamol"],
    )

    # Vision first, nothing discarded.
    assert merged[0] == "combiflam"
    assert set(merged) == {"combiflam", "ibuprofen", "paracetamol", "store", "tablets"}
    assert attribution["agreed"] == 1
    assert attribution["vision_only"] == 2
    assert attribution["ocr_only"] == 2


def test_merge_respects_the_cap():
    merged, _ = merge_tokens(
        ocr_tokens=[f"o{i}" for i in range(50)],
        vision_tokens=[f"v{i}" for i in range(50)],
        cap=20,
    )
    assert len(merged) == 20
    assert all(t.startswith("v") for t in merged)
