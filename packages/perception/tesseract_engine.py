"""
OCR over the DIP renditions, with per-word confidence.

`image_to_data` is used rather than `image_to_string` throughout. The plain
string API throws away the confidence Tesseract already computed, and confidence
is the single most useful signal this stage produces: it decides which rendition
won, which tokens are trustworthy enough to match a brand name against, and
whether the vision fallback is worth its cost. Discarding it and then trying to
guess quality from the text is strictly worse than reading the number.

Fusion across renditions is by **consensus vote**, not by naive union and not by
winner-takes-all. This was measured, not assumed. A plain union of every token
above the confidence floor performed *worse* than doing no preprocessing at all
on a degraded test image: it collected 63 words instead of 14, but the extra
tokens were `cong`, `cleves`, `fd`, `tobiets` — hallucinated readings from the
90-degree and 270-degree passes, which see horizontal text sideways and
confidently transcribe noise. Worse, the true brand token was pushed out.

Counting how many independent renditions produced each token separates the two
cases cleanly. A real word survives several different binarisations because it
is really there; a rendition-specific artefact does not, because nothing else
saw it. Tokens are therefore kept with their support count, and the resolver
weights by it.

Both views are exposed: `tokens` for recall (a token seen once may still be the
diagnostic one) and `consensus_tokens` for precision.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

try:
    import pytesseract
    from pytesseract import Output

    TESSERACT_AVAILABLE = True
except ImportError:  # pragma: no cover - import guard
    pytesseract = None
    Output = None
    TESSERACT_AVAILABLE = False


# Page segmentation modes worth trying. Tesseract's layout analysis assumes a
# document; a medicine strip is not one, and the wrong assumption costs more
# accuracy than any preprocessing gains back.
PSM_SPARSE = 11
"""Sparse text, no ordering. Best for scattered label fragments."""

PSM_BLOCK = 6
"""A single uniform block. Best for the composition paragraph."""

PSM_LINE = 7
"""A single text line. Used when a text-region crop is passed in."""

DEFAULT_PSMS = (PSM_SPARSE, PSM_BLOCK)

# Tesseract reports -1 for non-text blocks; anything under ~30 is noise in
# practice and admitting it corrupts the token bag the resolver matches against.
MIN_WORD_CONFIDENCE = 30.0

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-\+/\.]{1,}")
_STRENGTH_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|iu|%)\b", re.IGNORECASE)


@dataclass
class Word:
    """One recognised word, with where it came from."""

    text: str
    confidence: float
    left: int
    top: int
    width: int
    height: int
    rendition: str = ""
    rotation: int = 0
    support: int = 1
    """How many distinct renditions produced this token. The discriminator
    between a real word and a per-rendition hallucination."""


@dataclass
class OcrResult:
    """Fused OCR output across every rendition."""

    text: str
    """Text of the single best-scoring rendition — the most readable
    continuous transcription, used for display and as LLM context."""

    words: list[Word] = field(default_factory=list)
    """Every accepted word from every rendition, deduplicated."""

    tokens: list[str] = field(default_factory=list)
    """Full normalised token bag, ordered by support then confidence. Maximises
    recall — use when a single rare token could still be diagnostic."""

    consensus_tokens: list[str] = field(default_factory=list)
    """Tokens corroborated by more than one rendition, or read with high
    confidence by the winning one. Maximises precision — this is what the
    resolver matches on by default."""

    token_support: dict[str, int] = field(default_factory=dict)
    """Token -> number of renditions that produced it."""

    strengths: list[tuple[float, str]] = field(default_factory=list)
    """Parsed dose strengths, e.g. [(500.0, 'mg'), (125.0, 'mg')]. Highly
    diagnostic: '500mg + 125mg' pins Augmentin 625 far more sharply than the
    brand name does when the brand name is half torn off."""

    mean_confidence: float = 0.0
    best_rendition: str = ""
    rendition_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "tokens": self.tokens,
            "consensus_tokens": self.consensus_tokens,
            "strengths": [{"value": v, "unit": u} for v, u in self.strengths],
            "mean_confidence": round(self.mean_confidence, 2),
            "best_rendition": self.best_rendition,
            "word_count": len(self.words),
        }


def normalise_token(token: str) -> str:
    """Lowercase and strip punctuation from the edges of a token."""
    return token.strip(".,;:()[]{}/-+").lower()


def extract_tokens(text: str) -> list[str]:
    """Pull word-like tokens out of raw OCR text.

    Tokens must start with a letter and be at least two characters. Pure numbers
    are excluded here because batch numbers, dates and prices vastly outnumber
    meaningful numeric tokens and would swamp the match; dose strengths are
    recovered separately by `extract_strengths`, which keeps their unit attached
    and therefore their meaning.
    """
    seen: set[str] = set()
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text):
        token = normalise_token(match.group(0))
        if len(token) >= 2 and token not in seen:
            seen.add(token)
            tokens.append(token)
    return tokens


def extract_strengths(text: str) -> list[tuple[float, str]]:
    """Parse dose strengths such as '500 mg' or '125mg'."""
    out: list[tuple[float, str]] = []
    seen: set[tuple[float, str]] = set()
    for match in _STRENGTH_RE.finditer(text):
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        unit = match.group(2).lower()
        if (value, unit) not in seen:
            seen.add((value, unit))
            out.append((value, unit))
    return out


def _run_tesseract(image: np.ndarray, psm: int, lang: str) -> list[Word]:
    """One Tesseract pass. Returns accepted words."""
    if not TESSERACT_AVAILABLE:
        return []

    config = f"--oem 1 --psm {psm}"
    try:
        data = pytesseract.image_to_data(image, lang=lang, config=config, output_type=Output.DICT)
    except Exception:  # noqa: BLE001 - a failed pass must not sink the request
        return []

    words: list[Word] = []
    for i, raw in enumerate(data.get("text", [])):
        text = (raw or "").strip()
        if not text:
            continue
        try:
            confidence = float(data["conf"][i])
        except (ValueError, KeyError, IndexError):
            continue
        if confidence < MIN_WORD_CONFIDENCE:
            continue

        words.append(
            Word(
                text=text,
                confidence=confidence,
                left=int(data["left"][i]),
                top=int(data["top"][i]),
                width=int(data["width"][i]),
                height=int(data["height"][i]),
            )
        )
    return words


def _score_words(words: list[Word]) -> float:
    """Score a rendition's output.

    Weighted by word length as well as confidence, because Tesseract is often
    confident about single-character noise. A long word that is recognised at
    moderate confidence is far stronger evidence of real text than a short one
    at high confidence, and it is also far more useful for matching.
    """
    return sum(w.confidence * min(len(w.text), 12) for w in words) / 100.0


RENDITION_SCORE_FLOOR = 0.25
"""A rendition contributes tokens only if it scores at least this fraction of
the best rendition's score. See `read_renditions`."""


def read_renditions(
    renditions: list,
    *,
    lang: str = "eng",
    psms: tuple[int, ...] = DEFAULT_PSMS,
    score_floor: float = RENDITION_SCORE_FLOOR,
) -> OcrResult:
    """OCR every rendition and fuse the results by consensus.

    Accepts `Rendition` objects from `dip.pipeline` (anything with `.name`,
    `.image` and `.rotation`).

    Renditions scoring below `score_floor` times the best score are read but
    their tokens are **discarded**. This matters most for the rotated passes:
    the fan-out includes 90 and 270 degrees because Indian strips often print
    the composition vertically along an edge, but when the text is in fact
    horizontal those passes see everything sideways and return confident
    nonsense. Measured on a degraded test image, including them dropped token
    precision from 0.67 to 0.20. Gating on relative score keeps the rotated
    passes available for the case they exist for, while stopping them
    contributing when there is no vertical text to find — the ratio is the
    evidence, since a rendition that genuinely contains readable text scores
    within the same order of magnitude as the winner.
    """
    if not TESSERACT_AVAILABLE:
        return OcrResult(text="", mean_confidence=0.0, best_rendition="tesseract_unavailable")

    per_rendition: dict[str, list[Word]] = {}
    scores: dict[str, float] = {}
    texts: dict[str, str] = {}

    for rendition in renditions:
        best_for_this: list[Word] = []
        best_score = -1.0

        # Several page-segmentation modes per rendition. They disagree
        # substantially on packaging, and which one wins is not predictable from
        # the image — sparse mode finds scattered label fragments that block
        # mode merges into nonsense, while block mode reads the composition
        # paragraph that sparse mode fragments.
        # Upscaled renditions get one page-segmentation mode, not two. They are
        # the expensive passes (up to 12 MP against ~2 MP native), and PSM_BLOCK
        # is the right single choice for them: what upscaling exists to recover
        # is the composition paragraph, which is a uniform block of small print.
        # Halving their passes is a pure cost saving with no accuracy claim.
        rendition_psms = (PSM_BLOCK,) if rendition.name.startswith("up") else psms

        for psm in rendition_psms:
            words = _run_tesseract(rendition.image, psm, lang)
            score = _score_words(words)
            if score > best_score:
                best_score, best_for_this = score, words

        for word in best_for_this:
            word.rendition = rendition.name
            word.rotation = getattr(rendition, "rotation", 0)

        per_rendition[rendition.name] = best_for_this
        scores[rendition.name] = round(best_score, 2)
        texts[rendition.name] = " ".join(w.text for w in best_for_this)

    if not scores or all(not w for w in per_rendition.values()):
        return OcrResult(text="", mean_confidence=0.0, best_rendition="none",
                         rendition_scores=scores)

    best_rendition = max(scores, key=lambda k: scores[k])
    top_score = scores[best_rendition]

    # Drop renditions that scored far below the winner before fusing.
    threshold = top_score * score_floor
    accepted = [name for name, s in scores.items() if s >= threshold]
    all_words: list[Word] = [w for name in accepted for w in per_rendition[name]]

    if not all_words:
        return OcrResult(text="", mean_confidence=0.0, best_rendition=best_rendition,
                         rendition_scores=scores)

    # Deduplicate by normalised form, keeping the best reading of each token and
    # counting how many *distinct* renditions produced it.
    by_token: dict[str, Word] = {}
    supporters: dict[str, set[str]] = {}

    for word in all_words:
        key = normalise_token(word.text)
        if not key:
            continue
        supporters.setdefault(key, set()).add(word.rendition)
        if key not in by_token or word.confidence > by_token[key].confidence:
            by_token[key] = word

    for key, word in by_token.items():
        word.support = len(supporters[key])

    # Rank by support first, confidence second. Corroboration across independent
    # binarisations is stronger evidence than one model's self-reported
    # certainty — Tesseract is frequently confident about sideways noise.
    unique = sorted(by_token.values(), key=lambda w: (w.support, w.confidence), reverse=True)

    combined_text = " ".join(w.text for w in unique)
    token_support = {normalise_token(w.text): w.support for w in unique}

    # Tokens the winning rendition read. Always trusted, regardless of whether
    # anything else corroborates them.
    #
    # This exception exists because pure consensus voting actively destroys the
    # benefit of the best rendition. Adaptive upscaling exists precisely to read
    # small print that no other rendition can resolve — "amoxycillin" appears in
    # the upscaled pass and nowhere else — and a >=2 support rule then discards
    # exactly those tokens as uncorroborated. Measured: web-image accuracy went
    # from 7/16 to 6/16 when upscaled renditions were added under a pure
    # consensus rule, because the tokens they uniquely contributed were filtered
    # out. The winning rendition is the single most trustworthy source; other
    # renditions still need corroboration.
    best_tokens = {
        normalise_token(w.text) for w in per_rendition.get(best_rendition, []) if w.text
    }

    all_tokens = extract_tokens(combined_text)
    consensus = [
        t
        for t in all_tokens
        if token_support.get(t, 0) >= 2
        or t in best_tokens
        or (by_token.get(t) is not None and by_token[t].confidence >= 75.0)
    ]

    return OcrResult(
        text=texts.get(best_rendition, ""),
        words=unique,
        tokens=all_tokens,
        consensus_tokens=consensus,
        token_support={t: token_support.get(t, 1) for t in all_tokens},
        strengths=extract_strengths(combined_text),
        mean_confidence=float(np.mean([w.confidence for w in unique])),
        best_rendition=best_rendition,
        rendition_scores=scores,
    )


def available_languages() -> list[str]:
    """Installed Tesseract language packs.

    Hindi (`hin`) and other Indic packs matter here: Indian packaging is
    routinely bilingual, and running an English-only model over Devanagari
    produces confident garbage that then pollutes the token bag.
    """
    if not TESSERACT_AVAILABLE:
        return []
    try:
        return list(pytesseract.get_languages(config=""))
    except Exception:  # noqa: BLE001
        return []
