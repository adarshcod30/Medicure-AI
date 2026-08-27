# AWS setup for MediCure

You create the IAM user and paste two keys into `.env`. Everything else is
already wired.

## Why Amazon Nova and not Claude

Anthropic models on Bedrock are subscribed through **AWS Marketplace**. This
project's account is an **AISPL** account — Amazon Web Services India Private
Limited, billing in INR — and every Converse call to Claude returned:

    AccessDeniedException: Model access is denied due to
    INVALID_PAYMENT_INSTRUMENT. Your AWS Marketplace subscription for this
    model cannot be completed at this time.

...despite a valid Visa being on file and set as default. The blocker is the
Marketplace subscription path on the India reseller entity, not the card.

**Amazon Nova is first-party AWS.** No Marketplace subscription is involved, so
this class of failure does not apply. It is also multimodal, which the vision
transcription path needs, and cheaper than Claude for this workload.

| Role | Model | Used for |
|---|---|---|
| Primary | `us.amazon.nova-pro-v1:0` | Vision transcription, harder reasoning |
| Fast | `us.amazon.nova-lite-v1:0` | Explanation — rephrasing retrieved facts |
| Embeddings | `amazon.titan-embed-text-v2:0` | M3 vector search |

## Step 1 — create the IAM user

  https://console.aws.amazon.com/iam/home#/users

1. **Create user** → name `medicure-dev` → do **not** tick console access.
2. **Next** → *Attach policies directly* → **Create policy** (opens a new tab).
3. Choose the **JSON** tab, replace everything with the contents of
   `infra/aws/medicure-bedrock-policy.json`, and name it
   `MedicureBedrockAccess`.
4. Back in the user tab, refresh the policy list, tick `MedicureBedrockAccess`,
   and create the user.

The policy grants Nova and Titan invocation, model discovery and
`ApplyGuardrail`. Nothing else — no S3, no IAM, no ability to spend outside
Bedrock. If the keys leak, the blast radius is a Bedrock bill.

## Step 2 — create an access key

Open the user → **Security credentials** → **Create access key** → choose
*Application running outside AWS* → Create.

Copy both values now; the secret is shown once.

## Step 3 — enable model access

  https://console.aws.amazon.com/bedrock/home?region=us-east-1#/modelaccess

Enable **Nova Pro**, **Nova Lite**, and **Titan Text Embeddings V2**. Amazon's
own models are usually granted instantly.

Regional note: `us.` IDs are cross-region inference profiles routing across
us-east-1 / us-east-2 / us-west-2. If a model works intermittently, enable it
in all three.

## Step 4 — put the keys in .env

    cp .env.example .env

Then edit `.env` and fill in:

    AWS_REGION=us-east-1
    AWS_ACCESS_KEY_ID=AKIA...
    AWS_SECRET_ACCESS_KEY=...
    ENABLE_BEDROCK=true

`.env` is gitignored and will not be committed. Never paste the secret into a
chat, an issue, or a commit message.

**This works** — the app reads these explicitly and passes them to boto3.
pydantic-settings loads `.env` into the settings object *without* exporting to
`os.environ`, so boto3's default chain cannot see them; `BedrockClient` takes
them as arguments instead. `/v1/health` reports `credential_source: "env"` when
they are being used, so there is no ambiguity about which identity is active.

## Step 5 — verify

    bash infra/aws/verify.sh

Checks credentials, Bedrock reachability, actual invocation of both Nova
models, and Titan embeddings. It stops at the first real blocker and names the
specific fix.

Then:

    ENABLE_BEDROCK=true uvicorn apps.api.main:app --port 8000
    curl localhost:8000/v1/health | python3 -m json.tool

`capabilities.explanations` flips to `true` only after a real invocation
succeeds. A constructed client proves nothing — that was the whole lesson of
the Claude failure.
