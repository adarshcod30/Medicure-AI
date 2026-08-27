#!/usr/bin/env python3
"""
CDK app for the always-on MediCure deployment.

Infrastructure as code rather than console clicks, so the deployment is
reviewable and reproducible. One stack, one instance, no load balancer: an ALB
would roughly double the monthly cost to serve a workload that comfortably fits
on a single small box.

    pip install -r requirements.txt
    cdk deploy -c allowed_ssh_cidr=$(curl -s ifconfig.me)/32
"""

from __future__ import annotations

import aws_cdk as cdk

from medicure_stack import MedicureStack

app = cdk.App()

MedicureStack(
    app,
    "medicure-ai",
    description="MediCure AI - grounded medicine safety and affordability engine",
    env=cdk.Environment(
        account=app.node.try_get_context("account") or None,
        region=app.node.try_get_context("region") or "ap-south-1",
    ),
)

app.synth()
