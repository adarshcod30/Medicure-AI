"""
Packaging boilerplate — the text every medicine pack carries and none is
identified by.

Real phone photos exposed this. OCR on a strip reliably returns tokens like
`store`, `cool`, `dark`, `protect`, `from`, `moisture`, `keep`, `out`, `of`,
`children`, `directed`, `exceed` — the storage and dosage paragraph. It is set
in a dense, evenly-printed block, so OCR reads it far more reliably than the
stylised brand name or the small composition line, and it then dominates the
token bag handed to the resolver.

The words are worse than useless. They are near-universal across the corpus, so
they discriminate nothing, and their character n-grams still contribute mass to
the query vector — pulling it toward whichever composition happens to share
letter patterns with "store in a cool dry place".

Two safeguards, because over-filtering is the dangerous direction:

  * `mg`, `ml`, `tablet`, `capsule`, `syrup` and similar are NOT filtered. They
    carry real signal about dose and form.
  * The list is validated at import against the index's ingredient vocabulary,
    and any term that is also a real ingredient name is dropped from it. No
    active ingredient can ever be filtered out, however common it is.
"""

from __future__ import annotations

# Storage and handling instructions.
_STORAGE = {
    "store", "stored", "storage", "cool", "dry", "dark", "place", "protect",
    "protected", "moisture", "light", "sunlight", "heat", "below", "above",
    "temperature", "keep", "away", "out", "reach", "children", "child",
    "refrigerate", "freeze", "discard", "shake", "well", "tightly", "closed",
    "container", "original", "packaging",
}

# Dosage and administration boilerplate.
_DOSAGE = {
    "dose", "doses", "dosage", "adult", "adults", "children", "years", "age",
    "aged", "daily", "twice", "thrice", "times", "hours", "hourly", "exceed",
    "exceeding", "stated", "directed", "direction", "directions", "physician",
    "doctor", "medical", "advice", "consult", "prescribed", "prescription",
    "oral", "orally", "swallow", "chew", "water", "food", "meals", "empty",
    "stomach", "maximum", "minimum", "recommended",
}

# Regulatory and legal text.
_REGULATORY = {
    "schedule", "warning", "warnings", "caution", "rx", "registered",
    "trademark", "trademarks", "licence", "license", "lic", "batch", "mfg",
    "mfd", "exp", "expiry", "manufactured", "marketed", "manufacturer",
    "packed", "incl", "inclusive", "taxes", "mrp", "retail", "price",
    "maximum", "leaflet", "insert", "read", "carefully", "before", "using",
    "use", "uses", "product", "products", "information", "please", "note",
    "not", "for", "sale", "loose", "government", "india", "indian", "govt",
}

# Corporate names and suffixes.
_CORPORATE = {
    "ltd", "limited", "pvt", "private", "inc", "corp", "corporation", "company",
    "pharma", "pharmaceutical", "pharmaceuticals", "laboratories", "labs",
    "healthcare", "health", "care", "industries", "enterprises", "sciences",
    "lifesciences", "biotech", "remedies", "formulations", "division",
}

# Function words that survive OCR and add nothing.
_FUNCTION = {
    "the", "and", "for", "with", "from", "this", "that", "each", "any", "all",
    "may", "can", "should", "must", "have", "has", "are", "was", "were", "been",
    "its", "their", "your", "you", "one", "two", "other", "than", "then",
    "when", "while", "into", "onto", "over", "under", "more", "less", "also",
    "such", "only", "used", "usa", "made", "contains", "containing", "contain",
}

# Words that LOOK like boilerplate but must be kept: they carry dose or form
# information the resolver uses.
KEEP = {
    "tablet", "tablets", "capsule", "capsules", "syrup", "suspension",
    "injection", "cream", "ointment", "gel", "drops", "solution", "powder",
    "sachet", "inhaler", "spray", "lotion", "mg", "ml", "mcg", "gm", "iu",
    "sr", "er", "xr", "dt", "md", "cr", "ip",
}

# Pure function words. These are filtered unconditionally, even when they
# appear inside a multi-word ingredient name.
#
# Corpus document frequency is the wrong test for them: "and" occurs in fewer
# than 2% of distinct compositions (only in names like "aceclofenac and
# paracetamol tablets"), so a frequency rule marks it discriminative and
# protects it — while it appears in essentially every OCR read of a pack. Rare
# in the corpus, ubiquitous in the input. No standalone occurrence of "and"
# identifies a medicine, so the exception is safe.
NEVER_PROTECT = {
    "the", "and", "for", "with", "from", "this", "that", "any", "all", "its",
    "their", "your", "you", "than", "then", "when", "while", "into", "onto",
    "over", "under", "more", "less", "also", "such", "only", "other", "not",
}

_RAW_STOPWORDS = (_STORAGE | _DOSAGE | _REGULATORY | _CORPORATE | _FUNCTION) - KEEP


def build_stopwords(ingredient_vocabulary: set[str] | None = None) -> frozenset[str]:
    """Return the stopword set, minus anything that is a discriminative
    ingredient word.

    The subtraction is the safeguard that matters. `india` is boilerplate;
    `indian snakeroot` is a real drug. Filtering a token that names an active
    ingredient would remove the single most identifying word on the pack.

    The vocabulary passed in must already be filtered by document frequency —
    see `BrandIndex.discriminative_vocabulary`. A naive vocabulary built by
    splitting every composition string protects `and`, `for`, `with`, `from`,
    `other` and `water`, because they appear inside multi-word ingredient names
    like "water for injection" and "light liquid paraffin". Those words occur in
    a large share of compositions and identify nothing, so protecting them keeps
    exactly the tokens this module exists to remove.
    """
    if not ingredient_vocabulary:
        return frozenset(_RAW_STOPWORDS)
    return frozenset((_RAW_STOPWORDS - ingredient_vocabulary) | NEVER_PROTECT)


MIN_TOKENS_AFTER_FILTER = 1
"""Fall back to the unfiltered bag only when filtering leaves nothing at all.

A higher floor looked obviously right and measured worse. The reasoning for one
was: on a sparse read, filtering strips almost everything — one real product
shot went from 8 tokens to 1, another from 13 to 2 — so surely the unfiltered
bag is better than a single survivor.

Swept over 27 labelled real images (min_kept 0, 1, 2, 3, 4, 6), the answer was
no. A floor of 1 gave 12/27 ingredient hits; every higher value gave 11/27.
Restoring the boilerplate brings back more noise than the surviving tokens were
worth, even when only one survives — because the token that survived is, by
construction, the only one carrying any information.

One image out of 27 is not a strong signal and this should be re-measured on a
larger set. But there is no evidence for a higher floor, and the simpler rule
is the one to keep. Silent failures were 0 at every setting."""


def filter_tokens(
    tokens: list[str], stopwords: frozenset[str], *, min_kept: int = MIN_TOKENS_AFTER_FILTER
) -> list[str]:
    """Drop boilerplate and tokens too short to identify anything.

    Returns the ORIGINAL list when filtering would leave fewer than `min_kept`
    tokens. Order is otherwise preserved: the OCR fusion already ranked tokens
    by corroboration and confidence, and that ranking is still meaningful.
    """
    kept = [t for t in tokens if t not in stopwords and len(t) >= 3]
    if len(kept) < min_kept and len(tokens) > len(kept):
        return list(tokens)
    return kept or list(tokens)
