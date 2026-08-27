# EC2 deployment (CDK)

The always-on option. One `t4g.small` with an Elastic IP, running the Docker
Compose stack. Roughly ₹1,100–1,400/month in `ap-south-1` — the instance is
most of it, the EBS volume and Elastic IP are the rest.

Only worth doing when something must stay up without your laptop. For a demo,
[the Cloudflare tunnel](../../deploy/cloudflare-tunnel.md) is free and takes
five minutes.

## Prerequisites

```bash
npm install -g aws-cdk
```

```bash
pip install -r requirements.txt
```

Your IAM user needs permission to create EC2, IAM and CloudFormation
resources — broader than the scoped runtime policy in
`../medicure-bedrock-policy.json`, which is only what the *running service*
needs.

## Deploy

Bootstrap once per account and region:

```bash
cdk bootstrap
```

Then deploy, passing your own address so port 22 is not open to the world:

```bash
cdk deploy -c region=ap-south-1 --parameters allowedSshCidr=$(curl -s ifconfig.me)/32
```

Review what will be created before saying yes:

```bash
cdk diff -c region=ap-south-1
```

## After it comes up

The instance clones the repository and installs Docker, but does **not** start
the stack — it cannot, because `.env` still needs a real `JWT_SECRET` and, if
you want explanations, AWS credentials. Baking those into user data would put
them in the instance metadata service, readable by anything running on the box.

Connect (Session Manager needs no open port):

```bash
aws ssm start-session --target <instance-id>
```

Then:

```bash
cd /opt/medicure && sudo nano .env && sudo docker compose up -d --build
```

The first build takes several minutes — it installs Tesseract and OpenCV and
builds the index from `data/processed/`.

Check it:

```bash
curl -s http://<elastic-ip>:8000/v1/health | python3 -m json.tool
```

## Notes on the choices

- **No load balancer.** An ALB costs about as much as the instance, to serve a
  workload that fits on one box. Put Cloudflare in front for TLS instead.
- **No DocumentDB.** Atlas has a free tier that suits this better than a
  managed replica set for a personal medicine cabinet.
- **Graviton.** ~20% cheaper at equal memory, and every dependency here has an
  arm64 build.
- **2 GB, not 1.** Each uvicorn worker holds its own 125 MB index copy, which
  is why the Dockerfile runs a single worker; `t4g.micro` is too tight once
  OpenCV and Tesseract are loaded.

## Tearing it down

```bash
cdk destroy -c region=ap-south-1
```

Confirm the Elastic IP is released afterwards — an unassociated EIP is billed
hourly, which is a slow and annoying way to lose money on a project you thought
you had shut down.
