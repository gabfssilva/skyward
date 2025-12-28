"""AMI resolution for AWS EC2 instances.

Automatically selects the best public AMI based on region and GPU requirements.
Uses AWS SSM Parameter Store to get the latest AMI IDs.
"""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger("skyward.ami")

# SSM parameter paths for public AMIs
# CPU: Amazon Linux 2023 (lightweight, fast boot)
AL2023_CPU_SSM = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-6.1-x86_64"

# GPU: Deep Learning AMI Ubuntu 22.04 (has CUDA, cuDNN, NCCL pre-installed)
# Note: AL2023 DLAMI doesn't support G4dn/G5, so we use Ubuntu for better compatibility
DLAMI_GPU_SSM = (
    "/aws/service/deeplearning/ami/x86_64/"
    "base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"
)


@lru_cache(maxsize=32)
def resolve_ami(region: str, gpu: bool = False) -> str:
    """Resolve the best public AMI for the given configuration.

    Uses AWS SSM Parameter Store to get the latest AMI ID for the region.
    Results are cached to avoid repeated SSM calls.

    Args:
        region: AWS region (e.g., "us-east-1")
        gpu: Whether GPU support is needed (uses Deep Learning AMI if True)

    Returns:
        AMI ID string (e.g., "ami-0123456789abcdef0")

    Raises:
        RuntimeError: If AMI resolution fails
    """
    import boto3
    from botocore.exceptions import ClientError

    if gpu:
        param_name = DLAMI_GPU_SSM
        ami_type = "Deep Learning AMI (Ubuntu 22.04)"
    else:
        param_name = AL2023_CPU_SSM
        ami_type = "Amazon Linux 2023"

    logger.info(f"Resolving {ami_type} AMI for region {region}...")

    try:
        ssm = boto3.client("ssm", region_name=region)
        response = ssm.get_parameter(Name=param_name)
        ami_id: str = response["Parameter"]["Value"]
        logger.info(f"Resolved AMI: {ami_id}")
        return ami_id
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        if error_code == "ParameterNotFound":
            raise RuntimeError(
                f"Could not find {ami_type} AMI in region {region}. "
                f"SSM parameter {param_name} not found. "
                "Please specify an AMI explicitly using aws(ami='ami-xxx', region='...')"
            ) from e
        raise RuntimeError(
            f"Failed to resolve AMI in region {region}: {e}"
        ) from e


@lru_cache(maxsize=32)
def get_ami_username(region: str, ami_id: str) -> str:
    """Detect the default system username for an AMI.

    Queries AMI metadata to determine the correct username:
    - Ubuntu AMIs: 'ubuntu'
    - Amazon Linux / AL2023 AMIs: 'ec2-user'
    - Debian AMIs: 'admin'
    - RHEL/CentOS AMIs: 'ec2-user'

    Args:
        region: AWS region.
        ami_id: AMI ID to query.

    Returns:
        System username string.
    """
    import boto3
    from botocore.exceptions import ClientError

    try:
        ec2 = boto3.client("ec2", region_name=region)
        response = ec2.describe_images(ImageIds=[ami_id])

        if not response.get("Images"):
            logger.warning(f"AMI {ami_id} not found, defaulting to 'ec2-user'")
            return "ec2-user"

        image = response["Images"][0]
        name = (image.get("Name") or "").lower()
        description = (image.get("Description") or "").lower()

        # Check for Ubuntu
        if "ubuntu" in name or "ubuntu" in description:
            logger.info(f"AMI {ami_id} detected as Ubuntu, using 'ubuntu'")
            return "ubuntu"

        # Check for Debian
        if "debian" in name or "debian" in description:
            logger.info(f"AMI {ami_id} detected as Debian, using 'admin'")
            return "admin"

        # Default to ec2-user (Amazon Linux, RHEL, CentOS, etc.)
        logger.info(f"AMI {ami_id} defaulting to 'ec2-user'")
        return "ec2-user"

    except ClientError as e:
        logger.warning(f"Failed to query AMI {ami_id}: {e}, defaulting to 'ec2-user'")
        return "ec2-user"
