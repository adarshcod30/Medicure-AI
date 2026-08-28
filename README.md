# MediCure AI

**A grounded medicine safety and affordability engine for the Indian market.**

Photograph a medicine strip and MediCure tells you what it is, whether you are
being overcharged against the government ceiling price, and what cheaper
equivalent actually exists — **or it tells you it does not know.**

That last clause is the project.

```
DIP restores the image.  Retrieval identifies the drug.
The language model only explains what was retrieved.
```

---

## Why this is not a ChatGPT wrapper

"Photograph a strip → an AI explains the medicine" is something a frontier model
already does zero-shot. Building a pipeline for it adds nothing. So MediCure is
built around six things a language model provably cannot do alone.

| | Capability | Why a model alone cannot |
|---|---|---|
| **C0** | Digital image processing front-end | A model handed a warped, glare-blown blister cannot recover geometry or clipped highlights it never received. A homography and specular inpainting can. |
| **C1** | Calibrated abstention over 253,973 brands | A model cannot enumerate a proprietary namespace, and is confidently wrong rather than silent. |
| **C2** | Look-alike/sound-alike detection | Requires scanning the whole brand space for confusable neighbours. Thresholds set against the 99th percentile of 4,000 random name pairs. |
| **C3** | Deterministic NPPA price verification | Ceiling prices change by individual gazette order. No model knows them; asked, it invents them. |
| **C4** | Interaction and duplicate-therapy checks | Stateful and combinatorial across your persisted medicine cabinet. Findings come from DDInter rows with citations, never from a model. |
| **C5** | Groundedness enforcement + benchmark | The measurement is the contribution. |

### The headline number

150 held-out corrupted-OCR queries. **Both arms graded identically**, on
ingredient overlap — the raw LLM is Amazon Nova Pro on the same inputs with no
retrieval:

| | MediCure | Raw LLM |
|---|---|---|
| Accuracy (answering everything) | **88.7%** | 84.7% |
| Answered (coverage) | 78.7% | 95.3% |
| Precision when it answers | **99.2%** | 88.8% |
| Abstained | 21.3% | 4.7% |
| **Silent failure — confidently wrong** | **0.0%** | **10.7%** |
| Coverage at 95% precision | **90.7%** | 77.3% |
| Exact composition signature | 68.7% | n/a |
| Median latency | **62 ms** | 793 ms |

**Zero silent failures against 10.7%** is the result. Accuracy is close — the
model is good, and pretending otherwise would be dishonest — but it answers
95.3% of the time and is confidently wrong on roughly one query in nine, with
no way for a patient to tell which. MediCure declines 21.3% of the time and is
right on 99.2% of what it does answer.

**What that zero does and does not mean.** It is measured on synthetically
corrupted text, not on photographs. A user reported a real failure this
benchmark cannot see: a photographed Crocin strip, PARACETAMOL printed on the
foil, identified as `gelatin solutions` at 80% confidence. Read cleanly the
same text resolves correctly, so the failure needed an OCR pass that missed the
ingredient line and left packaging boilerplate behind. The gate added in
response requires a match's own composition or brand name to appear in the
query before it may be called confident (`match_is_corroborated`), and the
regression test pins it. But the honest claim is "zero on this benchmark",
not "zero in the field".

Two things stated plainly:

- **The LLM's ECE is better** (0.118 against 0.287). MediCure's is measured
  against the shared overlap rule while its calibrator was fitted to predict
  *exact signature* correctness, so the two are not measuring the same event.
  The calibration report has the figure against what it was actually trained on.
- **An earlier version of this table was wrong**, and in MediCure's favour it
  would have been easy to leave. MediCure was graded on exact signature match
  while the LLM was graded on token overlap — one arm had to be exactly right,
  the other only had to say "paracetamol" somewhere. Under that asymmetry the
  LLM appeared to *win* 84.7% to 68.7%. Comparing two systems means grading
  them the same way.

The severity breakdown is where the behaviour shows:

| Damage | Accuracy | Answered |
|---|---|---|
| Light | 97% | 98% |
| Moderate | 84% | 94% |
| **Heavy** | 42% | **56%** |

On severely damaged input it declines almost half the time — which is correct,
because on severely damaged input it is right only 42% of the time. The
coverage falls where the accuracy falls. That correspondence is what
calibration buys, and it is not available from a model that always answers.

Reproduce with `python -m eval.bench_identify --samples 300`, or add
`--with-llm-baseline` to run the no-retrieval Bedrock arm alongside it.

### And the number that matters more

On **12 real phone photographs** of Indian medicine packaging — one upside
down, four rotated 90°, two badly crumpled, two bilingual, one held in a hand
on a curved tin:

| | Phone photos | Web product shots |
|---|---|---|
| Images | 12 | 16 |
| Ingredient hit @1 | **8 / 12 (67%)** | **11 / 16 (69%)** |
| Answered | 10 / 12 | 9 / 16 |
| **Silent failure** | **0 / 12** | **0 / 16** |
| Vision transcription used | 10 / 12 | 14 / 16 |
| Seconds per image | 5.1 | 5.1 |

Vision transcription (Nova Pro, `--vision`) is what moved these: 13/28 to
**19/28** overall, with silent failure still at zero on this set.

Read that zero with the caveat above, and with one more: the vision path
samples, so these numbers move on their own. The same 28 images have scored
19, 20 and 21 across identical runs. A change worth ±1 here is not a result. The discipline in its
output is the interesting part — it returns `NostrosiI` with the OCR-style
capital-I typo *uncorrected*, and `stro-resistant` as the visible tail of
"gastro-resistant" *without completing it*. The model could trivially have
written both correctly. The prompt holds it to transcription, so the resolver
still decides what the drug is.

Scored separately, never pooled. Studio product photography is a far easier
distribution than a phone photo of a torn strip, and one averaged number would
describe neither.

Both confident answers were correct (cefixime + ofloxacin; beclomethasone +
neomycin). Every failure was an abstention, not a wrong answer given
confidently. That is the behaviour the whole architecture exists to produce,
and it only became measurable once real photographs existed.

Twelve images found three bugs that 1,200 synthetic queries did not — see
"Things this project got wrong". `python -m eval.bench_photos` reproduces it.

On **real retail photographs** (catalogue thumbnails, a different and easier
distribution), top-1 composition accuracy is **13.3%**, not 74%.

| | Synthetic corrupted text | Real retail photos |
|---|---|---|
| Top-1 correct | 74.3% | **13.3%** |
| Answered | 83.0% | 26.7% |
| Silent failure | 5.0% | 6.7% |

The gap is resolution, not algorithm. Retail thumbnails run 163×309 to 554×554,
where composition print is a few pixels tall; OCR returns `['ae','wey','be']`
and there is nothing to identify. The system abstains on 73% of them, which is
the correct response.

Publishing both numbers is the point. A synthetic benchmark alone would have
reported 74% and been wrong about the system's real behaviour by a factor of
five.

`python -m eval.bench_real_images` **cannot currently reproduce this.** Its
corpus lived in `data/raw/gimages`, which is gitignored and was therefore never
committed; it no longer exists on disk. The script needs `<dir>/harvest.tsv`
and `<dir>/images`, and the surviving photo sets are unlabelled, so they cannot
stand in. `eval/results/real_images.json` is kept, and marked stale in its own
`_provenance` block, as a record of what was run rather than a current result.
Reproducing it means re-harvesting retail listings — the labelling trick
(captions as free labels, described below) still works; only the images are
gone. For figures that do reproduce today, use `eval/results/photos.json` via
`python -m eval.bench_photos --orchestrator`.

---

## What it looks like working

```
POST /v1/search  {"query": "Crocin Advance Tablet"}

  identification   confident, P(correct) = 0.92
                   paracetamol 500mg  (263 products share this composition)

  price check      Rs 22.62 / 20 tablets = Rs 1.131 each
                   NPPA ceiling         = Rs 1.010 each
                   -> Rs 0.121 (12%) over, Rs 2.42 on this pack

  alternatives     Paracetamol Tablets IP 500 mg  (Jan Aushadhi)
                   Rs 0.70 per tablet — 38% cheaper
                   source: jan_aushadhi_pmbjp
```

And when it should not answer:

```
POST /v1/scan  <blurred, glare-covered photo>

  identification   unreadable
                   "severely out of focus; 98% of the image is blown-out glare.
                    Hold the phone steady and tap to focus. Turn the flash off
                    and shoot in indirect daylight."

  price_check      null
  alternatives     null
```

Those nulls are the architecture working. **There is no field in the response
schema for a fact that was not retrieved or computed**, so a hallucination has
nowhere to live.

---

## Quick start

```bash
git clone https://github.com/adarshcod30/Medicure-AI.git && cd Medicure-AI
```

```bash
docker compose up
```

The frontend is on http://localhost:5173, the API on http://localhost:8000/docs.
This runs **without an AWS account** — retrieval, price checks, alternatives and
abstention all work; only the natural-language explanation is omitted.

<details>
<summary>Running without Docker</summary>

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
brew install tesseract          # or: apt-get install tesseract-ocr
```

`tesseract` must be on PATH as a binary, not only as the `pytesseract` wrapper:
without it the orientation probes return no scores and `tests/test_dip.py`
fails in a way that looks like a code fault. Conda works equally well
(`conda create -n medicure-ai python=3.12`); the venv is spelled out because
`.venv/` is what `.gitignore` expects.

```bash
python scripts/build_index.py && python scripts/fit_calibrator.py
```

```bash
uvicorn apps.api.main:app --reload --port 8000
```

```bash
cd apps/web && npm install && npm run dev
```
</details>

---

## Architecture

```
React (Vite)
     |
FastAPI  ── one service; the Node/Express gateway was retired
     |
     ├─ perception     DIP pipeline → OCR ensemble → vision fallback (TRANSCRIBE ONLY)
     ├─ resolver       candidates → fuse → calibrate → abstain          [ML]
     ├─ grounding      Atlas Vector + Atlas Search → RRF                [RAG, M3]
     ├─ pharmacology   price · alternatives · interactions   [pure Python, no LLM]
     ├─ reasoning      Bedrock Converse + Guardrails                    [GenAI]
     └─ storage        MongoDB Atlas                                    [M3]
```

The ordering inside the pipeline is load-bearing, and three of the four orderings
were arrived at by watching it fail. They are documented at the top of
[`packages/perception/dip/pipeline.py`](packages/perception/dip/pipeline.py).

### The DIP layer

Thirteen modules under `packages/perception/dip/`, each stage independently
switchable so `eval/bench_ocr.py` can ablate them:

`acquire` · `denoise` (adaptive filter selection) · `enhance` (CLAHE, gamma,
unsharp) · `edges` (Canny with gradient-percentile hysteresis, Sobel, Scharr,
LoG) · `segment` (packet boundary, watershed, GrabCut) · `rectify` (4-point
homography, projection-profile deskew) · `glare` (specular inpainting,
multi-scale Retinex) · `morphology` (top-hat for embossed foil) · `binarize`
(Sauvola, Niblack, Wolf, Otsu) · `frequency` (homomorphic, FFT notch) ·
`textdetect` (MSER, stroke-width) · `quality` (the abstention gate) · `pipeline`

Inspect every stage on a real photo:

```bash
python -m packages.perception.dip.pipeline --image strip.jpg --dump-stages out/
```

---

## Things this project got wrong, and fixed

Kept here because the fixes are the interesting part, and each is a regression
test.

- **Auto-Canny's median rule returned almost nothing.** `upper = 1.33 × median`
  on a bright strip (median ≈ 238) clamps to 255 — above every gradient present
  — so the strong-edge set was empty. Thresholds now come from the gradient
  magnitude distribution.
- **Illumination normalisation erased the packet boundary.** It is a high-pass,
  and the packet/background step is low-frequency at its kernel size. Moved to
  after rectification.
- **Quality was measured after glare inpainting**, so a 12%-blown-out photo
  reported 0% glare. Since quality drives abstention, that inverted the central
  decision.
- **Multi-rendition OCR union was net-harmful**, cutting token precision from
  0.67 to 0.20 by admitting sideways hallucinations. Replaced with consensus
  voting plus a rendition score gate.
- **64.7% of the Jan Aushadhi catalogue lost its strength** because stripping the
  dosage form removed everything after it — and the majority convention puts the
  strength last. Generic substitution would have silently found nothing for
  two-thirds of the cheapest alternatives.
- **Doses were swapped between drugs** in `A and B Tablets 500mg + 125mg`.
- **`crocin 500` returned azithromycin** under a weighted score fusion, then
  again under a support bonus that rewarded market share over match quality.
- **Calibration did not transfer across input distributions.** Every feature
  described *match* quality; none described *query* quality. On a 242×208
  thumbnail whose OCR read `['ae','wey','be','tablets']`, one composition
  matched cleanly with a wide margin — the exact pattern that meant "correct"
  in training — so the calibrator returned **P = 0.93, confident, and wrong**.
- **180° rotation was never tried.** The fan-out used (0, 90, 270), so an
  upside-down strip was unreadable by construction. And the adaptive router
  *disabled* rotation for "good quality" images — four of five rotated real
  photos scored "good", because quality measures exposure and focus and says
  nothing about orientation. Tesseract's own OSD mode was tried first and
  failed on 10 of 12, reporting Fraktur and Cyrillic at confidence 0.4.
- **OCR read the storage paragraph, not the composition.** "Store in a cool dry
  place", "keep out of reach of children" is set in a dense even block that
  reads far better than a stylised brand name, and it crowded out the
  identifying tokens. 311 boilerplate tokens now filtered across 12 photos.

Full detail in the commit history.

---

## Known limitations

Stated here rather than discovered later.

- **NPPA ceilings cannot be cited.** `nppa_notif` and `nppa_date` are 0%
  populated in the source dataset, so a ceiling price cannot be traced to the
  gazette order that set it. `/v1/metrics` reports this. Closing it needs the
  NPPA gazette scrape — the agency publishes per-order PDFs, not a consolidated
  file.
- **Only 17.2% of products have a ceiling price at all.** Most medicines are not
  under price control. The system says "no ceiling on record" rather than
  implying fairness.
- **Only 11.5% of compositions have a Jan Aushadhi equivalent.** For the rest the
  honest answer is that none exists.
- **Calibration is fitted on synthetic corruption**, not real photographs. The
  thresholds are only as representative as the confusion model behind them, and
  need refitting against a real photo set.
- **The DIP ablation has not been run on real images.** On synthetic
  `cv2.putText` renders, full processing is *worse* than none on clean input
  (F1 0.70 vs 0.93) and better on degraded input (0.46 vs 0.21) — which is why
  the pipeline routes adaptively. Those numbers will change on real foil.

### The single highest-value thing you can contribute

**~300 photographs of real Indian medicine strips, labelled.** No public dataset
of Indian blister packs exists. Vary glare, tears, blur, angle, curvature and
partial occlusion deliberately — the ablation is only meaningful if the set
contains the failure modes. `eval/bench_ocr.py` is written and waiting.

---

## Tests

```bash
pytest -q
```

200 tests, and they need the environment above — every dependency, plus the
tesseract binary. The valuable ones warp an image by a known homography and assert
recovery, re-measure deskew output to catch a sign error that would silently
double the rotation, and assert that a composition signature is neither too
permissive (recommending a different drug) nor too strict (never finding the
generic).

---

## Deployment

| Option | Cost | Notes |
|---|---|---|
| Local + Cloudflare Tunnel | **$0** | Public HTTPS demo URL, no infrastructure |
| EC2 `t4g.small` (2 GB) | ~$12–15/mo | Correct sizing; ~$7/mo with a 1-year Savings Plan |
| AWS Free Tier `t3.micro` | $0/12mo | 1 GB RAM — too tight for the index plus OpenCV |

Frontend on S3 + CloudFront (~$1/mo). MongoDB Atlas M0 (free) is viable because
the 253k-row index lives in the application as an artifact, not in the database.
Bedrock is ~$2–6/month at 500 scans.

---

## Licence and disclaimer

MIT.

MediCure provides information, not medical advice. It is not a substitute for a
qualified healthcare professional. Always confirm with a pharmacist or doctor
before taking, changing or stopping any medicine.
