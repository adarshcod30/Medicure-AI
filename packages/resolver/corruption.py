"""
Synthetic OCR corruption — labelled training and evaluation data, for free.

The resolver's job is to recover a real brand name from a damaged reading of it.
Training and evaluating that needs (corrupted, true) pairs, and the obvious way
to get them is to photograph hundreds of strips and transcribe them by hand.
That work is still worth doing — real degradation has structure this cannot
reproduce — but it is not a prerequisite, because the corruption process itself
can be modelled.

The model is a **character confusion matrix** derived from how OCR actually
fails, not from uniform random noise. The distinction matters: uniform noise
would replace `m` with `q` as readily as with `rn`, and a matcher tuned against
that learns to be robust to errors that never happen while staying brittle to
the ones that do.

The confusions encoded here come from the standard failure modes of the
Tesseract/LSTM family on printed packaging:

  * **shape collisions** — `rn`/`m`, `cl`/`d`, `vv`/`w`. One glyph and a
    ligature of two occupy nearly the same ink.
  * **digit/letter collisions** — `0`/`O`, `1`/`l`/`I`, `5`/`S`, `8`/`B`,
    `2`/`Z`. Severe on pharmaceutical labels, which mix both freely.
  * **diacritic and stem loss** — `i`/`l`, `t`/`f` when print is faint.
  * **truncation** — torn strips and blister pockets covering the ends of
    words, which is a *positional* corruption, not a per-character one.

Truncation is modelled separately and deliberately biased to the leading edge,
because a strip is usually torn along a perforation at one end and the brand
name sits at the top.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# Substitution table. Values are the plausible misreadings of the key, in
# roughly descending order of how often OCR actually produces them.
CONFUSIONS: dict[str, tuple[str, ...]] = {
    # letter <-> letter, shape-based
    "m": ("rn", "nn", "in"),
    "rn": ("m",),
    "d": ("cl", "ci"),
    "cl": ("d",),
    "w": ("vv", "v"),
    "vv": ("w",),
    "n": ("h", "ri", "m"),
    "h": ("n", "b"),
    "u": ("v", "ii"),
    "v": ("u", "y"),
    "c": ("e", "o", "("),
    "e": ("c", "o", "a"),
    "a": ("o", "e", "@"),
    "o": ("0", "c", "e"),
    "i": ("l", "1", "j", "!"),
    "l": ("1", "i", "I", "t"),
    "t": ("f", "l", "+"),
    "f": ("t", "r"),
    "r": ("f", "n"),
    "s": ("5", "S", "$"),
    "g": ("9", "q", "8"),
    "b": ("6", "h", "8"),
    "y": ("v", "j"),
    "z": ("2", "s"),
    "k": ("lc", "x"),
    "p": ("q", "o"),
    "q": ("g", "p"),
    # digit <-> letter, the dominant class on medicine packaging
    "0": ("O", "o", "Q", "D"),
    "1": ("l", "I", "i", "7"),
    "2": ("Z", "z", "7"),
    "3": ("8", "B"),
    "5": ("S", "s", "6"),
    "6": ("b", "G", "5"),
    "7": ("1", "T", "/"),
    "8": ("B", "3", "6"),
    "9": ("g", "q", "0"),
    "O": ("0", "D", "Q"),
    "I": ("1", "l", "|"),
    "S": ("5", "$"),
    "B": ("8", "3"),
    "Z": ("2",),
    "G": ("6", "C"),
}

# Characters OCR spuriously inserts, usually from packaging texture, blister
# pocket edges or the boundary of an inpainted glare region.
NOISE_CHARS = tuple(".,'`|!:;-_~^\"")


@dataclass(frozen=True)
class CorruptionProfile:
    """How aggressively to damage a string.

    The three presets correspond to real capture conditions rather than to
    arbitrary severity levels, so evaluation results map onto something
    observable: `light` is a good photo of an intact strip, `heavy` is a torn
    strip under glare.
    """

    substitution_rate: float = 0.10
    deletion_rate: float = 0.03
    insertion_rate: float = 0.03
    case_flip_rate: float = 0.05
    space_damage_rate: float = 0.05
    truncate_probability: float = 0.0
    max_truncate_fraction: float = 0.35

    @classmethod
    def light(cls) -> CorruptionProfile:
        """A clear photo of an intact strip; OCR mostly works."""
        return cls(0.05, 0.01, 0.01, 0.05, 0.02, truncate_probability=0.0)

    @classmethod
    def moderate(cls) -> CorruptionProfile:
        """Typical handheld capture: some blur, some glare."""
        return cls(0.12, 0.04, 0.04, 0.10, 0.08, truncate_probability=0.15)

    @classmethod
    def heavy(cls) -> CorruptionProfile:
        """Torn strip, heavy glare, faint print. The case that matters."""
        return cls(0.25, 0.10, 0.08, 0.15, 0.15, truncate_probability=0.45,
                   max_truncate_fraction=0.5)


def corrupt(text: str, profile: CorruptionProfile, rng: random.Random) -> str:
    """Apply one sampled corruption to `text`."""
    if not text:
        return text

    chars = list(text)
    out: list[str] = []

    for char in chars:
        if char == " ":
            roll = rng.random()
            if roll < profile.space_damage_rate / 2:
                continue  # words run together
            if roll < profile.space_damage_rate:
                out.append("  ")  # word splits
                continue
            out.append(char)
            continue

        if rng.random() < profile.deletion_rate:
            continue

        emitted = char
        if rng.random() < profile.substitution_rate:
            options = CONFUSIONS.get(char) or CONFUSIONS.get(char.lower())
            if options:
                emitted = rng.choice(options)
                # Preserve the original case where the confusion produced a
                # single letter, so an uppercase label stays uppercase.
                if char.isupper() and len(emitted) == 1 and emitted.isalpha():
                    emitted = emitted.upper()

        if rng.random() < profile.case_flip_rate:
            emitted = emitted.lower() if emitted.isupper() else emitted.upper()

        out.append(emitted)

        if rng.random() < profile.insertion_rate:
            out.append(rng.choice(NOISE_CHARS))

    result = "".join(out)

    # Truncation last, so it removes already-corrupted characters rather than
    # protecting them.
    if result and rng.random() < profile.truncate_probability:
        result = _truncate(result, profile.max_truncate_fraction, rng)

    return result.strip()


def _truncate(text: str, max_fraction: float, rng: random.Random) -> str:
    """Remove characters from one end.

    Biased 2:1 toward the leading edge. Strips tear along the perforation at one
    end, and the brand name is printed at the top — so the front of the string
    is what physically goes missing, and a symmetric model would understate how
    hard the real case is.
    """
    length = len(text)
    if length <= 3:
        return text

    cut = rng.randint(1, max(1, int(length * max_fraction)))
    return text[cut:] if rng.random() < 0.67 else text[: length - cut]


def corrupt_many(
    text: str, profile: CorruptionProfile, count: int, *, seed: int = 0
) -> list[str]:
    """Generate several independent corruptions of one string."""
    rng = random.Random(seed)
    return [corrupt(text, profile, rng) for _ in range(count)]


def make_pairs(
    names: list[str],
    *,
    profiles: tuple[CorruptionProfile, ...] = (),
    per_profile: int = 1,
    seed: int = 0,
) -> list[tuple[str, str, str]]:
    """Build a labelled evaluation set.

    Returns `(corrupted, true_name, profile_name)` triples. The profile name is
    kept so results can be broken down by severity — an aggregate accuracy
    number hides whether a system is good on easy inputs and useless on hard
    ones, which is precisely the distinction that matters for abstention.
    """
    if not profiles:
        profiles = (
            CorruptionProfile.light(),
            CorruptionProfile.moderate(),
            CorruptionProfile.heavy(),
        )

    labels = {
        CorruptionProfile.light(): "light",
        CorruptionProfile.moderate(): "moderate",
        CorruptionProfile.heavy(): "heavy",
    }

    rng = random.Random(seed)
    pairs: list[tuple[str, str, str]] = []

    for name in names:
        for profile in profiles:
            label = labels.get(profile, "custom")
            for _ in range(per_profile):
                pairs.append((corrupt(name, profile, rng), name, label))

    return pairs
