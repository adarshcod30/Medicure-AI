# API keys — what each is for, and what it is not for

Every key in `.env`, why it exists, and whether it is currently wired in.

| Key | Status | Used for |
|---|---|---|
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | **live** | Bedrock Nova + Titan |
| `JWT_SECRET` | wired | Session tokens (auth lands with M3) |
| `GOOGLE_CLIENT_ID` / `_SECRET` | not yet | Google sign-in (M3) |
| `SARVAM_API_KEY` | **not yet** | Indic-language path — see below |
| `GEMINI_API_KEY` | **fallback only** | Contingency vision transcriber — see below |

## Amazon Bedrock — the primary, and verified

IAM user `medicure-dev` on account 104422508395. Verified working:

    us.amazon.nova-pro-v1:0        OK    vision + harder reasoning
    us.amazon.nova-lite-v1:0       OK    explanation
    us.amazon.nova-micro-v1:0      OK
    amazon.titan-embed-text-v2:0   OK    1024 dims, for M3 vector search

**Third-party Marketplace models fail on this account** with
`INVALID_PAYMENT_INSTRUMENT: Your AWS Marketplace subscription for this model
cannot be completed`, despite a valid card set as default. Those models are
delivered through AWS Marketplace, and this is an AISPL account — Amazon Web
Services India Private Limited, billing in INR — where that path is refused.
Amazon's own models are first-party and bypass Marketplace entirely, which is
why Nova works and the Marketplace ones do not. Not a billing fault; an
entity/channel one.

## Sarvam AI — the strongest unused key

Sarvam's models are built for Indian languages, and that maps onto a measured
gap. Indian packaging is routinely bilingual: **2 of the 12 real test photos
carry Devanagari alongside English** (`photo_03` an ORS sachet, `photo_05` an
Ayurvedic plaster). Running an English-only OCR pass over Devanagari does not
fail cleanly — it produces confident Latin-looking garbage, which then enters
the token bag the resolver matches against and actively degrades the result.

The intended use, in priority order:

1. **Script detection then routing.** If a region is Devanagari, do not send it
   to an English Tesseract model at all.
2. **Translation of Devanagari composition text to English** before matching,
   since the index is English-only. `बुखार` and `पेरासिटामोल` currently
   contribute nothing.
3. Possibly Indic ASR later, if a spoken-query interface is ever added.

Not wired in. It should be, and it is a better next perception step than more
DIP tuning, because it addresses input the pipeline currently cannot read *at
all* rather than input it reads imperfectly.

Do NOT use Sarvam to identify medicines or generate facts. Same rule as every
other model here: transcription and translation only, retrieval identifies.

## Google Gemini — deliberately demoted

The previous architecture used Gemini as its primary reasoning engine. It is
retained as a **fallback vision transcriber only**, and is not called today.

Two reasons it is not the primary:

1. This project's premise is Amazon Bedrock. A second vision provider is a
   contingency, not a design choice.
2. Gemini is what the replaced system used to *generate* medicine facts, prices
   and alternatives — the exact failure mode this architecture exists to
   prevent. Keeping it in a strictly narrower role is deliberate.

If it is ever enabled, it transcribes and nothing else. It must never populate
`identification`, `price_check` or `alternatives`.

## Rotating a key

`.env` is gitignored and has never been committed — verify with
`git log --all --full-history -- .env` (expect no output).

If an AWS key leaks: IAM → Users → `medicure-dev` → Security credentials →
deactivate, then delete, then create a new one. The attached policy grants only
Nova/Titan invocation, so the blast radius is a Bedrock bill, not your account.
