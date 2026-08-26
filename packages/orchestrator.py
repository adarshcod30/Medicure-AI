"""
The orchestrator — image or text in, grounded and cited answer out.

This is where the project's governing rule is enforced in code rather than
asserted in a prompt:

    DIP restores the image. Retrieval identifies the drug.
    The LLM only explains what was retrieved.

The consequence is structural. By the time any language model is involved, the
identification, the price arithmetic and the alternatives are already decided
and already carry their sources. The model is handed a filled-in result and
asked to phrase it. There is no field in the response object for a fact the
model originated, so there is nowhere for a hallucination to live.

Order of operations, and why:

  1. DIP restores the image and measures its quality
  2. Quality gate — a hopeless photo is refused HERE, before any spend
  3. OCR over the restored renditions, fused by consensus
  4. Retrieval ranks *compositions* against the token bag
  5. Calibration turns similarity into P(correct) and decides to answer or not
  6. Pharmacology computes price and alternatives — deterministic, no LLM
  7. Only then, optionally, the LLM explains

Steps 1-6 need no network and no AWS account. That is deliberate: everything
load-bearing is deterministic and testable, and disabling Bedrock costs the
system its prose, not its judgement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from packages.perception import tesseract_engine
from packages.perception.dip.pipeline import DipResult, run_auto
from packages.pharmacology.alternatives import AlternativesResult, find_alternatives
from packages.pharmacology.price import CeilingPriceTable, PriceCheck, check_price
from packages.resolver.calibrate import Calibrator
from packages.resolver.index import BrandIndex, BrandRecord, CompositionMatch

DISCLAIMER = (
    "This is information, not medical advice. Always confirm with a pharmacist "
    "or doctor before taking, changing or stopping any medicine."
)


@dataclass
class Identification:
    """What the system concluded, and how sure it is."""

    status: str
    """'confident', 'ambiguous', 'abstained' or 'unreadable'."""

    probability: float
    """Calibrated P(the top composition is correct). Meaningful only when
    `calibrated` is true — otherwise it is a raw similarity in disguise."""

    calibrated: bool
    composition: str | None = None
    signature: tuple = ()
    closest_brand: str | None = None
    brands_sharing_composition: int = 0
    candidates: list[dict] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "probability": round(self.probability, 4),
            "calibrated": self.calibrated,
            "composition": self.composition,
            "closest_brand": self.closest_brand,
            "brands_sharing_composition": self.brands_sharing_composition,
            "candidates_considered": self.candidates,
            "reason": self.reason,
        }


@dataclass
class ScanResult:
    """The full response contract. Every consequential field carries provenance."""

    identification: Identification
    image_quality: dict | None = None
    ocr: dict | None = None
    price_check: PriceCheck | None = None
    alternatives: AlternativesResult | None = None
    reference_product: BrandRecord | None = None
    explanation: dict | None = None
    elapsed_ms: float = 0.0
    stages: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "identification": self.identification.to_dict(),
            "image_quality": self.image_quality,
            "ocr": self.ocr,
            "reference_product": self.reference_product.to_dict()
            if self.reference_product
            else None,
            "price_check": self.price_check.to_dict() if self.price_check else None,
            "alternatives": self.alternatives.to_dict() if self.alternatives else None,
            "explanation": self.explanation,
            "disclaimer": DISCLAIMER,
            "timing_ms": {**self.stages, "total": round(self.elapsed_ms, 1)},
        }


class Orchestrator:
    """Wires the stages together. Construct once and share."""

    def __init__(
        self,
        index: BrandIndex,
        calibrator: Calibrator,
        ceiling_table: CeilingPriceTable,
        explainer: Any | None = None,
    ):
        self.index = index
        self.calibrator = calibrator
        self.ceiling_table = ceiling_table
        self.explainer = explainer
        """Optional. Anything with `.explain(ScanResult) -> dict`. Absent means
        no prose, and every other field is unaffected."""

    # --- entry points -----------------------------------------------------

    def analyse_image(self, image_bytes: bytes, *, explain: bool = True) -> ScanResult:
        started = time.perf_counter()
        stages: dict[str, float] = {}

        mark = time.perf_counter()
        dip: DipResult = run_auto(image_bytes)
        stages["dip"] = round((time.perf_counter() - mark) * 1000, 1)

        quality = dip.to_dict()

        # The quality gate fires before OCR, retrieval or any model call. A
        # photo this degraded cannot be identified by anything downstream, and
        # spending on it produces a confident-looking answer built on nothing.
        if dip.quality.should_abstain:
            return ScanResult(
                identification=Identification(
                    status="unreadable",
                    probability=0.0,
                    calibrated=self.calibrator.is_fitted,
                    reason=_retake_message(dip),
                ),
                image_quality=quality,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                stages=stages,
            )

        mark = time.perf_counter()
        ocr = tesseract_engine.read_renditions(dip.renditions)
        stages["ocr"] = round((time.perf_counter() - mark) * 1000, 1)

        # Consensus tokens are preferred, with the full bag as a fallback. When
        # only one rendition survived the score gate nothing can be corroborated
        # and consensus comes back empty — better a noisier query than none.
        tokens = ocr.consensus_tokens or ocr.tokens
        query = " ".join(tokens)

        if not query.strip():
            return ScanResult(
                identification=Identification(
                    status="unreadable",
                    probability=0.0,
                    calibrated=self.calibrator.is_fitted,
                    reason=(
                        "No readable text was found on this image. If this is a "
                        "medicine strip, try photographing the printed side in "
                        "better light."
                    ),
                ),
                image_quality=quality,
                ocr=ocr.to_dict(),
                elapsed_ms=(time.perf_counter() - started) * 1000,
                stages=stages,
            )

        result = self._resolve(query, strengths=ocr.strengths, stages=stages, started=started)
        result.image_quality = quality
        result.ocr = ocr.to_dict()

        if explain and self.explainer and result.identification.status != "abstained":
            result.explanation = self._explain(result, stages)

        result.elapsed_ms = (time.perf_counter() - started) * 1000
        return result

    def analyse_text(self, query: str, *, explain: bool = True) -> ScanResult:
        started = time.perf_counter()
        stages: dict[str, float] = {}

        result = self._resolve(query, strengths=(), stages=stages, started=started)

        if explain and self.explainer and result.identification.status != "abstained":
            result.explanation = self._explain(result, stages)

        result.elapsed_ms = (time.perf_counter() - started) * 1000
        return result

    # --- internals --------------------------------------------------------

    def _resolve(
        self, query: str, *, strengths, stages: dict, started: float
    ) -> ScanResult:
        mark = time.perf_counter()
        matches: list[CompositionMatch] = self.index.search_compositions_from_tokens(
            query.split(), strengths=strengths, top_k=5
        )
        stages["retrieval"] = round((time.perf_counter() - mark) * 1000, 1)

        status, probability = self.calibrator.decide(matches, query)

        identification = Identification(
            status=status,
            probability=probability,
            calibrated=self.calibrator.is_fitted,
            candidates=[m.to_dict() for m in matches[:5]],
        )

        if not matches:
            identification.reason = (
                "Nothing in the database of 253,973 Indian medicines matches this "
                "closely enough to name. It may be a product that is not in the "
                "dataset, or the text may have been misread."
            )
            return ScanResult(identification=identification, stages=stages)

        best = matches[0]
        identification.composition = best.label
        identification.signature = best.signature
        identification.closest_brand = best.best_name
        identification.brands_sharing_composition = best.support

        if status == "abstained":
            identification.reason = (
                f"The closest match is {best.label}, but confidence is only "
                f"{probability:.0%} — below the threshold this system will answer at. "
                "Rather than guess, please check with a pharmacist."
            )
            return ScanResult(identification=identification, stages=stages)

        if status == "ambiguous" and len(matches) > 1:
            identification.reason = (
                f"Two or more compositions match about equally well: {best.label} and "
                f"{matches[1].label}. These are different medicines, so the strip "
                "should be confirmed rather than assumed."
            )

        reference = self.index.record(best.best_row)

        mark = time.perf_counter()
        price = check_price(
            signature=best.signature,
            market_price=reference.price,
            pack_count=reference.pack_count,
            pack_unit=reference.pack_unit,
            table=self.ceiling_table,
        )
        alternatives = find_alternatives(
            self.index,
            signature=best.signature,
            reference_price_per_unit=reference.price_per_unit,
            reference_unit=reference.pack_unit,
            reference_form=reference.dosage_form,
            reference_row=reference.row,
        )
        stages["pharmacology"] = round((time.perf_counter() - mark) * 1000, 1)

        return ScanResult(
            identification=identification,
            reference_product=reference,
            price_check=price,
            alternatives=alternatives,
            stages=stages,
        )

    def _explain(self, result: ScanResult, stages: dict) -> dict | None:
        mark = time.perf_counter()
        try:
            explanation = self.explainer.explain(result)
        except Exception as exc:  # noqa: BLE001
            # An explanation failure must not take the answer down with it. The
            # facts are already computed and cited; prose is the optional layer.
            explanation = {
                "text": None,
                "error": f"explanation unavailable: {type(exc).__name__}",
            }
        stages["explanation"] = round((time.perf_counter() - mark) * 1000, 1)
        return explanation


def _retake_message(dip: DipResult) -> str:
    """Turn quality measurements into an instruction the user can act on.

    "I could not read this" is a dead end. "73% of this photo is blown-out
    glare — lay the strip flat and turn the flash off" is a retry.
    """
    reasons = "; ".join(dip.quality.reasons) or "the image is too degraded to read"
    advice = " ".join(dip.quality.advice)
    return f"This photo cannot be read reliably — {reasons}. {advice}".strip()
