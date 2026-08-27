"""
One EC2 instance, an Elastic IP, and the IAM to call Bedrock.

Deliberately modest. There is no load balancer, no autoscaling group and no
managed database, because none of them earn their cost here: the service is a
single stateless process in front of a read-only index, and MongoDB Atlas has a
free tier that is a better fit than a self-managed replica set on one box.

Notable choices:

- **Graviton (t4g).** ARM, ~20% cheaper than the x86 equivalent at the same
  memory. Every dependency in this project has an arm64 wheel or builds
  cleanly; the container base image is multi-arch.
- **2 GB RAM.** Each uvicorn worker loads its own copy of the 125 MB index, so
  the Dockerfile runs one worker. 2 GB leaves comfortable headroom for that
  plus OpenCV and Tesseract. t4g.micro (1 GB) does not.
- **Elastic IP.** Without one, the public address changes on every stop/start
  and any DNS record pointing at it silently breaks.
- **IMDSv2 required.** The default in modern CDK, but stated explicitly because
  IMDSv1's request-forgery exposure is exactly the sort of thing that gets
  inherited silently.
- **SSH restricted to a CIDR you pass in.** There is no default. An open port
  22 is the single most common way a demo instance becomes someone else's.
"""

from __future__ import annotations

import aws_cdk as cdk
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_iam as iam
from constructs import Construct


class MedicureStack(cdk.Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        allowed_ssh_cidr = cdk.CfnParameter(
            self,
            "allowedSshCidr",
            type="String",
            description=(
                "CIDR permitted to reach port 22. Use your own address, "
                "for example 203.0.113.4/32. Do not use 0.0.0.0/0."
            ),
            default="127.0.0.1/32",
            allowed_pattern=r"^(\d{1,3}\.){3}\d{1,3}/\d{1,2}$",
        )

        # The default VPC. A purpose-built VPC with private subnets would need
        # a NAT gateway, which costs more per month than the instance it exists
        # to serve.
        vpc = ec2.Vpc.from_lookup(self, "default-vpc", is_default=True)

        security_group = ec2.SecurityGroup(
            self,
            "medicure-sg",
            vpc=vpc,
            description="MediCure AI - HTTP, HTTPS, and restricted SSH",
            allow_all_outbound=True,
        )
        security_group.add_ingress_rule(
            ec2.Peer.any_ipv4(), ec2.Port.tcp(80), "HTTP from anywhere"
        )
        security_group.add_ingress_rule(
            ec2.Peer.any_ipv4(), ec2.Port.tcp(443), "HTTPS from anywhere"
        )
        security_group.add_ingress_rule(
            ec2.Peer.ipv4(allowed_ssh_cidr.value_as_string),
            ec2.Port.tcp(22),
            "SSH from the operator only",
        )

        role = iam.Role(
            self,
            "medicure-instance-role",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            description="MediCure AI instance role - Bedrock invoke and SSM access",
            managed_policies=[
                # Session Manager, so the instance is reachable without SSH at all.
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonSSMManagedInstanceCore"
                )
            ],
        )

        # Scoped to the first-party models this project actually uses. Anthropic,
        # Meta and Mistral models are delivered through AWS Marketplace, which
        # fails on an AISPL account without a Marketplace-valid payment
        # instrument - see apps/api/config.py.
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
                resources=[
                    f"arn:aws:bedrock:{self.region}::foundation-model/amazon.nova-*",
                    f"arn:aws:bedrock:{self.region}::foundation-model/amazon.titan-embed-*",
                    f"arn:aws:bedrock:*:{self.account}:inference-profile/*.amazon.nova-*",
                ],
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["bedrock:ApplyGuardrail"],
                resources=[f"arn:aws:bedrock:{self.region}:{self.account}:guardrail/*"],
            )
        )

        user_data = ec2.UserData.for_linux()
        user_data.add_commands(
            "set -eux",
            "dnf update -y",
            "dnf install -y docker git",
            "systemctl enable --now docker",
            "usermod -aG docker ec2-user",
            # Compose v2 as a CLI plugin; the standalone docker-compose binary
            # is not packaged for Amazon Linux 2023.
            "mkdir -p /usr/local/lib/docker/cli-plugins",
            "curl -fsSL "
            "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-aarch64 "
            "-o /usr/local/lib/docker/cli-plugins/docker-compose",
            "chmod +x /usr/local/lib/docker/cli-plugins/docker-compose",
            # The repository is public; a private one needs a deploy key here.
            "cd /opt && git clone https://github.com/adarshcod30/Medicure-AI.git medicure",
            "cd /opt/medicure && cp .env.example .env",
            # The stack cannot start usefully until .env holds a real JWT_SECRET
            # and, if explanations are wanted, AWS credentials. Deliberately not
            # baked into user data, which is readable from the instance metadata
            # service by anything running on the box.
            "echo 'MediCure cloned. Edit /opt/medicure/.env, then: "
            "docker compose up -d --build' > /etc/motd",
        )

        instance = ec2.Instance(
            self,
            "medicure-instance",
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            instance_type=ec2.InstanceType.of(
                ec2.InstanceClass.BURSTABLE4_GRAVITON, ec2.InstanceSize.SMALL
            ),
            machine_image=ec2.MachineImage.latest_amazon_linux2023(
                cpu_type=ec2.AmazonLinuxCpuType.ARM_64
            ),
            security_group=security_group,
            role=role,
            user_data=user_data,
            require_imdsv2=True,
            block_devices=[
                ec2.BlockDevice(
                    device_name="/dev/xvda",
                    # 30 GB: the image carries the 126 MB index plus OpenCV and
                    # Tesseract, and Docker layer churn during rebuilds needs
                    # room the 8 GB default does not leave.
                    volume=ec2.BlockDeviceVolume.ebs(
                        30, encrypted=True, volume_type=ec2.EbsDeviceVolumeType.GP3
                    ),
                )
            ],
        )

        address = ec2.CfnEIP(self, "medicure-eip", domain="vpc")
        ec2.CfnEIPAssociation(
            self,
            "medicure-eip-association",
            allocation_id=address.attr_allocation_id,
            instance_id=instance.instance_id,
        )

        cdk.CfnOutput(self, "publicIp", value=address.ref, description="Elastic IP")
        cdk.CfnOutput(
            self,
            "sshCommand",
            value=f"ssh ec2-user@{address.ref}",
            description="SSH (or use Session Manager, which needs no open port)",
        )
        cdk.CfnOutput(
            self,
            "healthUrl",
            value=f"http://{address.ref}:8000/v1/health",
            description="Capability report - check this first after deploying",
        )
