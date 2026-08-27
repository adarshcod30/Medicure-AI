# Deploying the frontend to Vercel

Free, and a good fit: `apps/web` is a Vite build that produces static assets.

The API is a separate deployment — see [cloud-run.md](cloud-run.md). Deploy the
API **first**, because the frontend needs its URL baked in at build time.

## Why the config exists

`apps/web/vercel.json` does two things worth knowing about:

- **SPA rewrites.** The app has client-side routes (`/login`, `/history`,
  `/cabinet`). Without a rewrite, a visitor who refreshes on `/history` gets a
  404 from Vercel's static host, because no such file exists — React Router
  never gets a chance to run. The rewrite sends everything except `/assets/*`
  to `index.html`. The `assets/` exclusion matters: rewriting those too would
  serve HTML in place of missing JS and produce a confusing MIME-type error
  instead of an honest 404.
- **Immutable asset caching.** Vite fingerprints filenames
  (`index-BHpmphCf.js`), so those files can never change under a given name and
  are safe to cache for a year.

## Deploy

From the repository root:

```bash
npx vercel --cwd apps/web
```

Vercel will ask to link a project the first time. Accept the detected Vite
framework; `vercel.json` supplies the rest.

Set the API URL — **it is read at build time, not runtime**, so it must be set
before the production build:

```bash
npx vercel env add VITE_API_URL production --cwd apps/web
```

Enter your Cloud Run URL with the `/v1` suffix, for example
`https://medicure-api-xxxxx.a.run.app/v1`.

Then ship it:

```bash
npx vercel --prod --cwd apps/web
```

## Point the API back at it

CORS is an explicit allow-list, not a wildcard — the API issues session tokens,
so a wildcard would defeat the purpose. Add your Vercel domain and redeploy the
API:

```bash
gcloud run services update medicure-api --region asia-south1 --update-env-vars "CORS_ORIGINS=https://your-app.vercel.app,http://localhost:5173"
```

## Verify

Open the Vercel URL and check, in order:

1. The dashboard loads and a text search returns a result.
2. Refresh directly on `/history` — it should render, not 404. That is the
   rewrite working.
3. The browser console shows no CORS errors. If it does, `CORS_ORIGINS` does
   not contain your exact Vercel origin (scheme included).
4. The first API call may take 15–40s while Cloud Run cold-starts. The axios
   client allows 120s for this.

## A note on what is public

The frontend is a static bundle, so **everything in it is readable by anyone**.
That is fine here: `VITE_API_URL` is a public endpoint, and no key is baked in.
Do not add one — the AWS credentials, JWT secret and Mongo URI live only in the
API's Secret Manager entries, and nothing in `apps/web` should ever need them.
