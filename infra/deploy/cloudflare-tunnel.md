# Publishing MediCure with a Cloudflare Tunnel

The zero-cost path. The API keeps running on your machine; Cloudflare gives it
a public HTTPS URL and terminates TLS. Nothing is billed, no ports are opened
on your router, and your home IP is never exposed.

This is the right choice for a demo, a viva, or sharing a link with someone for
an afternoon. It is not the right choice for something that must stay up when
your laptop sleeps — see [README.md](README.md) for that.

## Install

```bash
brew install cloudflared
```

## The 30-second version

Start the API:

```bash
conda activate medicure-ai && PYTHONPATH=$PWD python -m uvicorn apps.api.main:app --port 8000
```

Then, in a second terminal:

```bash
cloudflared tunnel --url http://localhost:8000
```

`cloudflared` prints a `https://<random-words>.trycloudflare.com` URL. That URL
is live immediately and needs no Cloudflare account.

Check it end to end — `/v1/health` reports which capabilities are actually up,
so it is the honest thing to look at first:

```bash
curl -s https://<your-tunnel>.trycloudflare.com/v1/health | python -m json.tool
```

**The URL changes every restart**, and the tunnel dies with the terminal. For a
stable address, use a named tunnel below.

## Pointing the frontend at it

The frontend reads its API base from `VITE_API_URL` at build time, so it must
be set before `npm run build`:

```bash
VITE_API_URL=https://<your-tunnel>.trycloudflare.com/v1 npm run build --prefix apps/web
```

Serve the built assets however you like — a second quick tunnel works:

```bash
npx serve apps/web/dist -l 5174
```

```bash
cloudflared tunnel --url http://localhost:5174
```

Then add the frontend's tunnel URL to `CORS_ORIGINS` in `.env` and restart the
API. CORS is enforced against an explicit list; a wildcard would defeat the
purpose given the API now issues session tokens.

## A stable URL (named tunnel)

Needs a free Cloudflare account and a domain on it.

```bash
cloudflared tunnel login
```

```bash
cloudflared tunnel create medicure
```

That writes a credentials JSON and prints a tunnel UUID. Create
`~/.cloudflared/config.yml`:

```yaml
tunnel: <TUNNEL-UUID>
credentials-file: /Users/<you>/.cloudflared/<TUNNEL-UUID>.json

ingress:
  - hostname: medicure.<your-domain>
    service: http://localhost:8000
  - service: http_status:404
```

Route DNS at it, then run it:

```bash
cloudflared tunnel route dns medicure medicure.<your-domain>
```

```bash
cloudflared tunnel run medicure
```

To keep it alive across reboots, install it as a service:

```bash
sudo cloudflared service install
```

## What to check before sharing a link

- `ENVIRONMENT=production` in `.env`, so error responses stop including
  internals.
- `JWT_SECRET` is a real generated secret, not `change-me`. Anyone who knows
  the default can mint tokens for any account.
- `CORS_ORIGINS` lists your frontend's real origin and nothing else.
- `STORE_UPLOADS=false` unless you have a reason — the project's claim that it
  does not retain medical images should stay true.
- `/v1/health` shows the capabilities you expect. If `accounts` is false your
  `MONGODB_URI` is not reachable from this process; if `explanations` is false
  Bedrock has not completed a successful call yet.

## Known rough edges

- Quick tunnels are rate-limited and occasionally slow to establish. If the
  URL 502s for the first few seconds, that is the tunnel connecting, not the
  API failing — `curl localhost:8000/v1/health` to confirm.
- A cold start loads a 125 MB index, so the first request after boot takes a
  few seconds. The axios client already allows a 120s timeout for this.
- Image uploads travel over the tunnel; a 12 MB photo on a slow uplink is the
  slowest part of a scan by a wide margin.
