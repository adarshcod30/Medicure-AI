# MediCure AI — working notes

## The one rule

    DIP restores the image. Retrieval identifies the drug.
    The LLM only explains what was retrieved.

Enforced structurally, not by prompting. There is no field in the response
schema for a fact the model originated. Before adding one, ask what stops it
being fabricated.

## Non-negotiables

- **Never let a language model produce a price, a brand name, a dose or an
  interaction.** Those come from `packages/pharmacology/` and
  `packages/resolver/`, which contain no model calls at all.
- **An empty answer is a valid answer.** `find_alternatives` returning `[]` and
  `identification.status == "abstained"` are correct outcomes, not bugs. The
  system this replaced had a prompt saying "NEVER leave the list empty", which
  is what guarantees invention.
- **Every claim carries provenance.** `source: {dataset, record_id, url}`.
- **Set `maxTokens` on every Bedrock call.** Unset, it reserves the model
  maximum against quota and causes throttling at low request rates.
- **Amazon Nova, not Claude.** Anthropic models are subscribed via AWS
  Marketplace and fail on this AISPL (AWS India) account with
  INVALID_PAYMENT_INSTRUMENT despite a valid card. Nova is first-party and
  multimodal. `bash infra/aws/probe_models.sh` re-checks empirically.
- **The vision model transcribes; it never identifies.** It can name the drug
  and would usually be right — that is exactly why it is refused. See
  `packages/perception/vision_transcribe.py`.
- **Decide with measurements.** Several defaults here (fusion rule, name
  weight, DIP preset routing, contrast metric) were chosen by benchmark after
  the intuitive choice measured worse. Add to `eval/` rather than arguing.

## Layout

```
apps/api/            FastAPI — the only backend
apps/web/            React (Vite)
packages/
  perception/dip/    13 DIP modules, each switchable via DipConfig
  perception/        tesseract_engine (consensus fusion)
  resolver/          normalize · corruption · index · calibrate
  pharmacology/      price · alternatives          <- no LLM in here, ever
  reasoning/         bedrock · explainer
  orchestrator.py    wires the stages
eval/                benchmarks; the differentiation lives here
scripts/             build_index · fit_calibrator
```

## After changing anything

```bash
pytest -q                                    # 132 tests
python scripts/build_index.py                # if normalize.py changed
python scripts/fit_calibrator.py             # if index or features changed
python -m eval.bench_identify --samples 300  # if the resolver changed
```

Changing `packages/resolver/normalize.py` invalidates the index *and* the
calibrator — composition signatures shift, so a stale index silently produces
wrong matches. `ARTIFACT_VERSION` guards the index; bump it when the layout
changes.

## Environment

```bash
conda activate medicure-ai
PYTHONPATH=$PWD python -m pytest tests/ -q
```

OpenCV is pinned `>=4.10,<5`. OpenCV 5 changed `HoughLinesP`'s return shape
from `(N,1,4)` to `(N,4)`; the code handles both, but 5.x is not otherwise
exercised.

## Known gaps (do not rediscover these)

- NPPA gazette notifications are 0% populated — ceilings cannot be cited.
- 17.2% ceiling coverage, 11.5% Jan Aushadhi coverage. Both are real, both are
  reported by `/v1/metrics`, neither is a bug.
- Calibration is fitted on synthetic corruption, not photographs.
- 28 labelled real images exist (12 phone captures, 16 web shots) under
  `data/raw/`, gitignored. `python -m eval.bench_photos --vision` is the real
  benchmark. At n=28 it is too small to tune against — differences of one or
  two images are noise, and two "obvious" improvements measured worse.
- `photo_07` is unlabelled; nobody has identified it.
