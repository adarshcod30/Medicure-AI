# Deploying the API to Google Cloud Run

The always-on free option. Cloud Run scales to zero, so you pay only while a
request is running, and the monthly free tier covers far more than this project
will ever serve.

Pairs with **Vercel** for the frontend and **MongoDB Atlas M0** for storage,
both free. Total expected cost: **about ₹11/month**, all of it Artifact
Registry image storage.

## The one setting that decides your bill

| `--min-instances` | behaviour | cost |
|---|---|---|
| **0** | scales to zero; billed only during requests | **~₹11/month** |
| 1 | one container warm 24/7 | **~$70/month** |

At `min-instances=0` the free tier — 180,000 vCPU-seconds and 360,000
GiB-seconds per month — works out to 50 hours of billed container time. A scan
takes about 5 seconds, so that is **roughly 36,000 scans a month, free**.

At `min-instances=1` you are billed for 2 GiB and 1 vCPU every second of the
month whether anyone visits or not. That is the $70.

The trade for staying at 0 is cold starts. **Do not "fix" a slow first request
by raising this value** — warm it instead (see below).

`deploy-cloud-run.sh` hardcodes `MIN_INSTANCES=0` so the expensive setting is
not something you can reach by accident.

## The two settings that decide whether it crashes

Measured on this repository, not guessed:

```
after the index loads              806 MB resident
peak during one image scan       1,536 MB
```

- **`--memory 2Gi`.** 1 GiB OOMs on every photo scan, which is the main use
  case. 2 GiB leaves ~512 MB of headroom.
- **`--concurrency 1`.** Cloud Run defaults to **80** concurrent requests per
  instance. Each in-flight scan costs ~730 MB on top of the shared 806 MB
  index, so two at once need ~2.3 GB and the instance is killed. Concurrency
  here is bought with memory; one request per instance is the honest setting.

`--max-instances 3` bounds the blast radius. Cloud Run's default is 100, and a
crawler or a retry loop can fan out to all of them before you notice.

## One-time setup

Authenticate (this needs a browser, so it is yours to run):

```bash
gcloud auth login
```

Find or create a project — **do not deploy into an unrelated one**:

```bash
gcloud projects list
```

```bash
gcloud projects create medicure-ai-prod --name="MediCure AI"
```

Cloud Run requires billing enabled even to use the free tier. Link it in the
console, then set a budget so a mistake cannot run away:

**Billing → Budgets & alerts → Create budget → ₹100/month, alert at 50% and
100%.** This is worth the two minutes. A budget alert is the only thing that
tells you about a misconfiguration before the invoice does.

## Secrets

Never pass these with `--set-env-vars`; they end up readable in the revision
config. Put them in Secret Manager once:

```bash
printf '%s' "$(python3 -c 'import secrets;print(secrets.token_urlsafe(48))')" | gcloud secrets create medicure-jwt-secret --data-file=-
```

```bash
printf '%s' 'mongodb+srv://USER:PASS@cluster.mongodb.net/medicure' | gcloud secrets create medicure-mongodb-uri --data-file=-
```

Repeat for `medicure-aws-key-id`, `medicure-aws-secret`, and
`medicure-guardrail-id`. Then grant the runtime service account access:

```bash
gcloud projects add-iam-policy-binding PROJECT_ID --member="serviceAccount:$(gcloud projects describe PROJECT_ID --format='value(projectNumber)')-compute@developer.gserviceaccount.com" --role=roles/secretmanager.secretAccessor
```

## Deploy

```bash
bash infra/deploy/deploy-cloud-run.sh YOUR_PROJECT_ID
```

It enables the required APIs, creates the Artifact Registry repository, builds
with Cloud Build, and deploys with every measured setting above.

## Verify

```bash
curl -s https://YOUR-SERVICE-URL/v1/health | python3 -m json.tool
```

The first request after idle is a cold start: pulling a ~1.8 GB image,
importing ~190 MB of OpenCV/sklearn/scipy, then loading the index. Expect
**15–40 seconds**. The frontend already allows a 120s timeout for exactly this.

Confirm it really will scale to zero:

```bash
gcloud run services describe medicure-api --region asia-south1 --format='value(spec.template.metadata.annotations)' | tr ',' '\n' | grep -i instances
```

You want `minScale: '0'`.

## Cold starts, honestly

`--cpu-boost` is on, which helps. Beyond that:

- For a demo or viva, use the [Cloudflare tunnel](cloudflare-tunnel.md)
  instead — instant, no cold start, and free.
- For a link in a CV or README, Cloud Run is right: a visitor waiting 20
  seconds once is fine, and it costs nothing while nobody visits.
- Warm it before a demo by hitting `/v1/health` a minute beforehand.

## What this does not cover

`data/artifacts/` is built inside the image at build time, so a rebuild is
needed whenever `normalize.py` or the catalogues change. The dense vectors
(`build_embeddings.py`) are **not** built in the image — they need AWS
credentials — but dense retrieval is off by default, so nothing depends on
them.
