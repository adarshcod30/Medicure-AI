# AWS setup for MediCure

Four steps, in this order. Step 1 is the actual blocker — the other three are
quick and none of them will work until it is done.

## 1. Add a payment method  ← the blocker

`INVALID_PAYMENT_INSTRUMENT` is an **account billing state**, not a Bedrock
permission and not a model-access setting. Every Converse call is refused until
the account has a valid payment method, no matter what IAM says.

  https://console.aws.amazon.com/billing/home#/paymentpreferences

Add a card and set it as default. Wait ~2 minutes; the error message says so
explicitly and it is accurate.

Cost for this project is small: a scan is ~3-6k input / 800 output tokens, so
500 scans/month on a Haiku+Sonnet mix is roughly **$2-6/month**. Embedding a
20k-chunk corpus with Titan is a one-time ~$0.10.

## 2. Confirm Bedrock model access

  https://console.aws.amazon.com/bedrock/home?region=us-east-1#/modelaccess

Enable, at minimum:
- Claude Sonnet 4.6
- Claude Haiku 4.5
- Titan Text Embeddings V2  (needed for M3 vector search)

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
