# Deploying MediCure AI

Three paths, in the order most people should try them.

| path | cost | always on | setup | when it fits |
|---|---|---|---|---|
| **Cloudflare Tunnel** | ₹0 | no — dies with your terminal | ~5 min | demos, a viva, sharing a link for an afternoon |
| **Cloud Run + Vercel** | ~₹11/mo | yes, scales to zero | ~30 min | a link in a CV or README |
| **Docker Compose** | ₹0 local | while the host is up | ~15 min + image build | a spare machine, a lab box, a Pi you leave on |
| **EC2 t4g.small (CDK)** | ~₹1,100–1,400/mo | yes, always warm | ~30 min | when cold starts are unacceptable |

Start at the top. The tunnel is genuinely enough for showing the project to
someone, and it costs nothing to abandon. Cloud Run is the one to put in a CV:
free at this traffic, always reachable, at the price of a 15-40s cold start.

- [cloudflare-tunnel.md](cloudflare-tunnel.md) — the ₹0 demo path
- [cloud-run.md](cloud-run.md) — the API, always on and free
- [vercel.md](vercel.md) — the frontend, free
- [../aws/cdk/README.md](../aws/cdk/README.md) — EC2, when warm matters

## What has to travel with the app

Two directories are gitignored and are **not** in a fresh clone:

- `data/artifacts/` (~126 MB) — the lexical index. Rebuildable:
  `python scripts/build_index.py`
- `data/processed/` (~34 MB) — the source catalogues the index is built from.

The `Dockerfile` handles this by copying `data/processed/` in and running
`scripts/build_index.py` **at image build time**. That is deliberate: fitting
TF-IDF over 253,973 rows takes ~15 seconds, and doing it per container start
makes every deploy and every restart pay for it. The trade is a larger image
and a slower build, which is the right way round.

If you enable dense retrieval (off by default — it measured worse, see
`eval/bench_dense.py`), `scripts/build_embeddings.py` needs AWS credentials, so
it cannot run at image build time in CI without them. Run it separately and
mount `data/artifacts/`.

## 1. Cloudflare Tunnel

See [cloudflare-tunnel.md](cloudflare-tunnel.md).

## 2. Docker Compose

From the repository root:

```bash
docker compose up --build
```

That brings up three services: the API on `:8000`, MongoDB on `:27017`, and the
Vite dev server on `:5173`. Bedrock is off by default in the compose file, so
the stack runs with no AWS account at all — retrieval, price verification,
alternatives, LASA and abstention all work, and only the natural-language
explanation is missing. That degradation is worth being able to demonstrate.

To enable explanations, set your credentials in `.env` and flip
`ENABLE_BEDROCK: "true"` in `docker-compose.yml`.

To run without the bundled Mongo (for Atlas), point `MONGODB_URI` at your
cluster and stop the `mongo` service.

## 3. EC2 with CDK

See [../aws/cdk/README.md](../aws/cdk/README.md). Infrastructure-as-code rather
than console clicking, per the repository's AWS guidance.

The instance is `t4g.small` — Graviton, 2 GB RAM. That sizing is not arbitrary:
each uvicorn worker loads its own copy of the 125 MB index, which is why the
Dockerfile runs a single worker and why a smaller instance is not comfortable.

## Verifying any deployment

`/v1/health` is the honest endpoint. It reports each capability separately and
names what is degraded and why:

```bash
curl -s https://<your-host>/v1/health | python -m json.tool
```

`explanations` stays `false` until a Bedrock call has actually **succeeded** —
a constructed client proves nothing, because an account without a valid payment
instrument builds one fine and then fails every call.

`accounts`, `scan_history` and `medicine_cabinet` are all false when MongoDB is
unreachable. A deployment can be fully useful with all three false; the
frontend renders that as a capability notice rather than an error.

## Before exposing anything publicly

- `JWT_SECRET` must not be the default. Anyone who knows it can mint a token
  for any account.
- `ENVIRONMENT=production`.
- `CORS_ORIGINS` names your real frontend origin only.
- `STORE_UPLOADS=false` keeps the no-retention claim true.
- Keep the Bedrock IAM policy scoped to Nova and Titan
  (`infra/aws/medicure-bedrock-policy.json`). Detach
  `medicure-setup-policy.json` once the guardrail exists.
