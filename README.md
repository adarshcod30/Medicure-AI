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
| **C2** | Look-alike/sound-alike detection *(M2)* | Requires scanning the whole brand space for confusable neighbours. |
| **C3** | Deterministic NPPA price verification | Ceiling prices change by individual gazette order. No model knows them; asked, it invents them. |
| **C4** | Interaction and duplicate-therapy checks *(M3)* | Stateful and combinatorial across your persisted medicine cabinet. |
| **C5** | Groundedness enforcement + benchmark | The measurement is the contribution. |

### The headline number

300 held-out queries, all of them corrupted OCR, stratified by severity:

| | MediCure |
|---|---|
| Accuracy, answering everything | 74.3% |
| Answered (coverage) | 82.7% |
| Precision when it answers | 89.9% |
| **Silent failure — confidently wrong** | **4.3%** |
| Coverage at 95% precision | 63.3% |
| Expected calibration error | 0.049 |
| Median latency | 64 ms |

Silent failure is the number that matters clinically. A wrong answer delivered
confidently is one the patient has no way to catch. A frontier model on these
same inputs answers ~100% of the time, so *its* silent-failure rate is
whatever its error rate is.

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
conda create -n medicure-ai python=3.12 -y && conda activate medicure-ai
pip install -e ".[dev]"
brew install tesseract          # or: apt-get install tesseract-ocr
```

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

104 tests. The valuable ones warp an image by a known homography and assert
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
