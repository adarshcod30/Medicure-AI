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

from packages.perception import boilerplate, tesseract_engine, vision_transcribe
from packages.perception.dip.pipeline import DipResult, run_auto
from packages.pharmacology.alternatives import AlternativesResult, find_alternatives
from packages.pharmacology.price import CeilingPriceTable, PriceCheck, check_price
from packages.resolver.calibrate import Calibrator
from packages.resolver.index import BrandIndex, BrandRecord, CompositionMatch
from packages.resolver.normalize import canonical_ingredient

DISCLAIMER = (
    "This is information, not medical advice. Always confirm with a pharmacist "
    "or doctor before taking, changing or stopping any medicine."
)

def _ingredient_vocabulary(index: BrandIndex) -> set[str]:
    """Every word used in any active-ingredient name in the catalogue."""
    words: set[str] = set()
    for signature in index._signatures:  # noqa: SLF001 - same package family
        for component in signature or ():
            name = canonical_ingredient(str(component[0]))
            if name:
                words.update(name.split())
    words.discard("")
    return words


def _better(candidate, current) -> bool:
    """Is `candidate` a strictly more useful identification than `current`?

    Status first, probability only as a tie-break within the same status. A
    confident answer beats an ambiguous one regardless of the numbers, because
    the statuses mean different things to the reader; but two abstentions are
    ranked by which one the calibrator liked more.
    """
    a, b = _rank(candidate.status), _rank(current.status)
    if a != b:
        return a > b
    return candidate.probability > current.probability


def _rank(status: str) -> int:
    """Order identification statuses by how much they actually tell the user.

    Used to decide whether a vision rescue improved anything. `unreadable` and
    `abstained` are both refusals, but `abstained` at least names a closest
    match, so it sits above.
    """
    return {"unreadable": 0, "abstained": 1, "ambiguous": 2, "confident": 3}.get(status, 0)


COVERAGE_CAVEAT = (
    "Two things cause this: the photo may be too damaged to read, or the product "
    "may not be in the catalogue at all. The catalogue covers prescription "
    "medicines, and does NOT include most vitamins, supplements and OTC health "
    "products — brands like Becosules, Revital, Zincovit and Shelcal are absent "
    "entirely, so they can never be identified here no matter how clear the photo."
)
"""Said out loud on every abstention, because the alternative is worse.

A crumpled Becosules strip abstained at 1% and reported only "confidence is too
low", which reads as "try a better photo". No photo would have worked:
`becosules` returns 0 of 253,973 brands. The A-Z dataset is prescription
pharmaceuticals, and nutraceuticals were never in it.

Telling someone to retake a photo that cannot succeed is a worse failure than
saying "not in the database" — it wastes their time and implies the system is
closer to an answer than it is."""


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
            # JSON-safe list-of-lists, the same shape signature_to_json in
            # packages/storage/mongo.py produces — the cabinet accepts this
            # form back verbatim, so the identity round-trips bit-for-bit.
            "signature": [list(component) for component in self.signature],
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
    vision: dict | None = None
    """Vision transcription, when it fired. Text and provenance only — there is
    no field here for an identification the model produced."""
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
            "vision": self.vision,
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
        transcriber: Any | None = None,
        dense_reranker: Any | None = None,
    ):
        self.index = index
        self.calibrator = calibrator
        self.ceiling_table = ceiling_table
        self.explainer = explainer
        self.transcriber = transcriber
        self.dense_reranker = dense_reranker
        """Optional embedding-based reranker over the lexical candidates. It
        reorders, never introduces — see resolver/dense.py — and any failure
        inside it degrades to the lexical ranking. Enabled by measurement
        (eval/bench_dense.py), not by existing."""
        """Optional vision transcriber. Fires only when the quality gate says
        the image is degraded, and produces TEXT ONLY — it never identifies."""
        """Optional. Anything with `.explain(ScanResult) -> dict`. Absent means
        no prose, and every other field is unaffected."""

        # Built once from the index's own discriminative vocabulary, so no
        # active ingredient can ever be filtered out as boilerplate.
        self._stopwords = boilerplate.build_stopwords(index.discriminative_vocabulary())

        # Every word appearing in any active-ingredient name the catalogue
        # knows. A closed vocabulary, used to narrow a query that drowned.
        self._ingredient_words = _ingredient_vocabulary(index)

    def _narrow(self, tokens: list[str]) -> list[str]:
        """Keep only tokens that name a known active ingredient.

        The blunt instrument that fixes a measured failure. A sharp photo of a
        Crocin strip yields 122 tokens, because OCR reads the whole package
        insert — dosage, contraindications, the manufacturer's address — and
        every one of those words is RARE in a corpus of drug names, so
        `boilerplate` scores them discriminative and keeps them. The one token
        that mattered, 'paracetamol', was outvoted 121 to 1 and retrieval
        returned "gynaecological products".

        Narrowing to the ingredient vocabulary cut 122 tokens to 4 and returned
        paracetamol 500mg. Vision transcription of the same strip went 96 -> 5
        with the same answer, which is the useful part: two independent readers
        converge once the prose is gone.

        This is deliberately a SECOND attempt, never the first. The full bag
        carries brand names, and brands are not in this vocabulary — narrowing
        unconditionally would throw away the strongest signal on every strip
        whose brand name reads cleanly."""
        return [
            t for t in tokens
            if t in self._ingredient_words
            or canonical_ingredient(t) in self._ingredient_words
        ]

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
        #
        # ...unless a vision model is available, in which case the gate was
        # contradicting itself. `assess` sets should_abstain and
        # use_vision_fallback from the same severity score, so an "unusable"
        # photo was simultaneously marked "worth a vision call" and refused
        # before that call could happen. Measured: seven of the twenty-eight
        # labelled images returned `unreadable` this way, and Tesseract had
        # already recovered 'meropenem' from one of them before the refusal
        # discarded it.
        #
        # The gate still holds when there is no transcriber: without one there
        # is genuinely nothing further to try, and refusing early saves the OCR
        # fan-out on a hopeless image.
        if dip.quality.should_abstain and self.transcriber is None:
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

        # Strip packaging boilerplate. On real photographs OCR reliably returns
        # the storage and dosage paragraph — "store in a cool dry place", "keep
        # out of reach of children" — because it is set in a dense even block
        # that reads far better than a stylised brand name. Those words appear
        # on every pack, identify nothing, and crowd out the composition tokens.
        # filter_tokens returns the original bag when filtering would strip it
        # below a usable size — see MIN_TOKENS_AFTER_FILTER.
        tokens = boilerplate.filter_tokens(tokens, self._stopwords)

        # --- vision transcription -----------------------------------------
        # Fires only when the quality gate judged the image degraded or worse.
        # That is where Tesseract starts dropping characters and where the
        # extra cost is justified; a clean photo does not need it.
        #
        # It returns TEXT, never an identification. Its tokens join the same
        # bag, go through the same resolver and the same calibration. Measured
        # on a crumpled Combiflam strip: Tesseract produced ['by','the','store',
        # 'mg','away','adults'] while vision produced ['sanofi','combiflam',
        # 'ibuprofen','paracetamol'] — brand and composition.
        vision_info: dict | None = None
        if self.transcriber and dip.quality.use_vision_fallback:
            mark = time.perf_counter()
            transcription = self.transcriber.transcribe(dip.processed)
            stages["vision"] = round((time.perf_counter() - mark) * 1000, 1)

            if transcription.available and transcription.tokens:
                vision_tokens = boilerplate.filter_tokens(
                    transcription.tokens, self._stopwords
                )
                tokens, attribution = vision_transcribe.merge_tokens(tokens, vision_tokens)
                vision_info = {**transcription.to_dict(), "attribution": attribution}
            else:
                vision_info = transcription.to_dict()

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

        # --- second chance A: narrow the query -----------------------------
        # Free (no model call), so it is tried first. See _narrow for why the
        # full bag can lose to four of its own tokens.
        if result.identification.status in {"abstained", "ambiguous"}:
            narrowed = self._narrow(tokens)
            if narrowed and len(narrowed) < len(tokens):
                candidate = self._resolve(
                    " ".join(narrowed),
                    strengths=ocr.strengths,
                    stages=stages,
                    started=started,
                )
                if _better(candidate.identification, result.identification):
                    result = candidate
                    stages["narrowed_tokens"] = len(narrowed)

        # --- second chance B: escalate on a weak RESULT, not a weak image ---
        #
        # The quality gate above asks "does this photograph look degraded?".
        # That is not the same question as "did OCR read anything useful?", and
        # a measured failure showed how far apart they can be.
        #
        # A sharp, evenly lit Crocin Advance strip scored verdict="good", so
        # vision never fired. Tesseract read the dense instructions block and
        # the manufacturer's address perfectly — 'every', 'days', 'without',
        # 'drugs', 'patiala', 'gsk', 'licensed' — and missed the large
        # "PARACETAMOL FAST RELEASE TABLETS" entirely. Those boilerplate tokens
        # retrieved "gynaecological products", and the system abstained on a
        # photograph a human reads at a glance. Rotating the same strip by 90
        # degrees changed which text won and produced the right answer, which is
        # how obvious it was that the *image* was never the problem.
        #
        # So: if the cheap path did not reach an answer, spend the vision call
        # even on a good-looking image. The cost profile is right — one extra
        # model call only when the free path has already failed — and vision
        # still only ever returns TEXT, through the same resolver and the same
        # calibration.
        if (
            self.transcriber
            and vision_info is None
            and result.identification.status in {"abstained", "ambiguous"}
        ):
            mark = time.perf_counter()
            transcription = self.transcriber.transcribe(dip.processed)
            stages["vision_rescue"] = round((time.perf_counter() - mark) * 1000, 1)

            if transcription.available and transcription.tokens:
                vision_tokens = boilerplate.filter_tokens(
                    transcription.tokens, self._stopwords
                )
                merged, attribution = vision_transcribe.merge_tokens(tokens, vision_tokens)
                # Three readings of the same strip, best one wins: everything
                # merged, the vision text alone, and the vision text narrowed to
                # ingredient names. On the Crocin strip the merged bag still
                # lost — 60 tokens of insert prose outvoting the drug name —
                # while the narrowed bag resolved it in five.
                rescued = None
                for bag in (merged, vision_tokens, self._narrow(vision_tokens)):
                    if not bag:
                        continue
                    candidate = self._resolve(
                        " ".join(bag),
                        strengths=ocr.strengths,
                        stages=stages,
                        started=started,
                    )
                    if rescued is None or _better(candidate.identification, rescued.identification):
                        rescued = candidate
                if rescued is None:
                    rescued = result
                vision_info = {
                    **transcription.to_dict(),
                    "attribution": attribution,
                    "trigger": "weak_result",
                }
                # Keep the rescue only if it actually improved the verdict.
                # Vision text is not automatically better than Tesseract text,
                # and quietly replacing a result with a worse one to justify the
                # call would be its own kind of dishonesty.
                if _better(rescued.identification, result.identification):
                    result = rescued
                else:
                    vision_info["discarded"] = "did not improve the identification"
            else:
                vision_info = {**transcription.to_dict(), "trigger": "weak_result"}

        result.image_quality = quality
        result.ocr = ocr.to_dict()
        result.vision = vision_info

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

        if self.dense_reranker is not None and matches:
            mark = time.perf_counter()
            matches = self.dense_reranker.rerank(query, matches, top_k=5)
            stages["dense_rerank"] = round((time.perf_counter() - mark) * 1000, 1)

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
                "closely enough to name. " + COVERAGE_CAVEAT
            )
            return ScanResult(identification=identification, stages=stages)

        best = matches[0]
        identification.composition = best.label
        identification.signature = best.signature
        identification.closest_brand = best.best_name
        identification.brands_sharing_composition = best.support

        if status == "abstained":
            identification.reason = (
                f"This could not be identified. The closest thing in the database is "
                f"{best.label}, but at {probability:.0%} confidence that is far more "
                "likely to be a coincidental text match than the right answer — so it "
                "is not being offered as one. "
                + COVERAGE_CAVEAT
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
