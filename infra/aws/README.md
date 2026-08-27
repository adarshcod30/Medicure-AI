# AWS setup for MediCure

Four steps, in this order. Step 1 is the actual blocker — the other three are
quick and none of them will work until it is done.

## 0. First, find out what actually works

    bash infra/aws/probe_models.sh

Listing a model proves nothing — an account with no payment instrument lists
every model and invokes none of them. The probe calls each candidate for real.

**An IAM user does not fix INVALID_PAYMENT_INSTRUMENT.** That error is a
property of the *account*, not of the identity calling it, so a new IAM user on
the same account hits the identical wall.

But the model family matters, and the error message says why:

> Your **AWS Marketplace subscription** for this model cannot be completed.

Anthropic, Meta, Mistral and Cohere models on Bedrock are delivered through AWS
Marketplace subscriptions, and it is the Marketplace path that requires a
payment instrument. **Amazon Nova and Titan are first-party AWS services and do
not go through Marketplace at all.** They may work on an account where Claude
does not — which is why the defaults are now Nova.

If the probe shows Nova working, you can skip step 1 entirely and set the
access keys as described in step 4.

## 1. Add a payment method — only if the probe shows nothing working

`INVALID_PAYMENT_INSTRUMENT` is an **account billing state**, not a Bedrock
permission and not a model-access setting.

  https://console.aws.amazon.com/billing/home#/paymentpreferences

Add a card and set it as default. Wait ~2 minutes; the error message says so
explicitly and it is accurate.

Cost for this project is small: a scan is ~3-6k input / 800 output tokens, so
500 scans/month on a Haiku+Sonnet mix is roughly **$2-6/month**. Embedding a
20k-chunk corpus with Titan is a one-time ~$0.10.

## 2. Confirm Bedrock model access

  https://console.aws.amazon.com/bedrock/home?region=us-east-1#/modelaccess

Enable, at minimum:
- Amazon Nova Pro and Nova Lite  (first-party; the current defaults)
- Titan Text Embeddings V2       (needed for M3 vector search)

Claude, Llama and Mistral are optional. They are better models, but they route
through Marketplace and so depend on billing being resolved.

**Regional gotcha:** the `us.` model IDs are *cross-region inference profiles*
that route across us-east-1, us-east-2 and us-west-2. Access granted in only one
of those can still fail when the profile routes elsewhere. Enable in all three
if a model works intermittently.

## 3. Stop using root

The current identity is `arn:aws:iam::104422508395:root`. Root has no
restrictions, cannot be scoped, and its keys cannot be safely rotated.

  https://console.aws.amazon.com/iam/home#/users

1. Create user `medicure-dev`, no console access.
2. Attach a customer-managed policy from `medicure-bedrock-policy.json` in this
   directory. It grants Claude and Titan invocation and nothing else — no S3,
   no IAM, no ability to spend outside Bedrock.
3. Create an access key, choosing "Application running outside AWS".

## 4. Configure credentials locally

Run this yourself — never paste a secret key into a chat, a file, or a commit:

    aws configure --profile medicure

Region `us-east-1`, output `json`. Then:

    export AWS_PROFILE=medicure

Or put the keys in `.env` instead, which the app reads directly:

    AWS_ACCESS_KEY_ID=...
    AWS_SECRET_ACCESS_KEY=...
    AWS_REGION=us-east-1

`.env` is gitignored. Never commit it, and never paste a secret key into a chat
or an issue.

`aws login` also works but issues **short-lived** credentials that expire
mid-session — which is what has been happening. A configured profile persists.

## Verify

    bash infra/aws/verify.sh

It checks credentials, Bedrock reachability, actual invocation of both models,
and Titan embeddings — stopping at the first real blocker so the output names
one thing to fix rather than a wall of red.

## Then

    ENABLE_BEDROCK=true uvicorn apps.api.main:app --port 8000
    curl localhost:8000/v1/health

`capabilities.explanations` flips to `true` only after a real invocation
succeeds — a constructed client proves nothing, since an account without a
payment method builds one happily and then fails every call.


## Which model to use

The defaults are Amazon Nova. Set `BEDROCK_MODEL_ID` and `BEDROCK_FAST_MODEL_ID`
in `.env` to whatever the probe reports as working.

Model choice matters less here than in most systems, and that is by design. The
model never produces a fact — it rephrases a fact sheet that retrieval and
arithmetic have already filled in, and a Guardrail checks its output against
that sheet. A weaker model produces a clumsier sentence, not a wrong price.

Swapping families costs one line because the client uses the **Converse API**,
which presents a single request shape across every provider. Moving the whole
system from Claude to Nova touched one default parameter and two config lines.
Under `InvokeModel` it would have meant rewriting the request body for a
different provider schema.
