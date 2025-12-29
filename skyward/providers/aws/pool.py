"""EC2 Instance Pool - manages stopped instances for fast startup."""

from __future__ import annotations

import base64
import uuid
from dataclasses import dataclass
from functools import cached_property
from math import ceil
from typing import TYPE_CHECKING, Any

from botocore.exceptions import ClientError
from tenacity import (
    RetryError,
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
)

from skyward.constants import (
    DEFAULT_INSTANCE_NAME,
    INSTANCE_RUNNING_MAX_ATTEMPTS,
    INSTANCE_RUNNING_WAIT_DELAY,
    SSM_WAIT_SECONDS,
    UV_INSTALL_URL,
    InstanceState,
    SkywardTag,
)
from skyward.spec import (
    AllocationStrategy,
    Spot,
    SpotCapacityError,
    SpotStrategy,
    _SpotNever,
    normalize_spot,
)


def _is_spot_capacity_error(exc: BaseException) -> bool:
    """Check if exception is a retriable spot capacity error."""
    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code", "")
        return code in ("InsufficientInstanceCapacity", "SpotMaxPriceTooLow")
    return False


class FleetCapacityError(Exception):
    """Raised when fleet doesn't get enough instances (triggers retry)."""

    def __init__(self, requested: int, got: int) -> None:
        super().__init__(f"Fleet got {got}/{requested} instances")
        self.requested = requested
        self.got = got


def _is_fleet_capacity_error(exc: BaseException) -> bool:
    """Check if exception is a retriable fleet capacity error."""
    return isinstance(exc, FleetCapacityError)

if TYPE_CHECKING:
    from mypy_boto3_ec2 import EC2Client
    from mypy_boto3_ec2.type_defs import DescribeInstancesResultTypeDef

    from skyward.providers.aws.infra import AWSResources
    from skyward.providers.aws.ssm import SSMSession
    from skyward.volume import Volume


def _generate_volume_script(
    volumes: tuple[Volume, ...],
    username: str = "ec2-user",
) -> str:
    """Generate shell script to install and mount volumes.

    Args:
        volumes: Tuple of Volume objects to mount.
        username: Username for tilde expansion (e.g., "ubuntu" -> /home/ubuntu).

    Returns:
        Shell script as a string. Empty string if no volumes.
    """
    if not volumes:
        return ""

    # Determine home directory for tilde expansion
    # root user has /root, others have /home/username
    if username == "root":
        home_dir = "/root"
    else:
        home_dir = f"/home/{username}"

    lines = ["# Install and mount volumes"]

    # Collect all install commands (deduplicated)
    install_cmds: set[str] = set()
    for vol in volumes:
        install_cmds.update(vol.install_commands())

    for cmd in install_cmds:
        lines.append(cmd)

    # Get uid/gid for the user (will be resolved at runtime via id command)
    # We inject shell commands to get the actual uid/gid
    lines.append(f'MOUNT_UID=$(id -u {username})')
    lines.append(f'MOUNT_GID=$(id -g {username})')

    # Mount each volume (with tilde and uid/gid expansion)
    for vol in volumes:
        for cmd in vol.mount_commands():
            # Expand ALL occurrences of ~ and ~/ to user's home directory
            # This handles both standalone ~ at start AND ~ inside strings (like fstab entries)
            expanded_cmd = cmd.replace("~/", f"{home_dir}/")
            # Also handle standalone ~ (e.g., "mkdir -p ~")
            if expanded_cmd == "~":
                expanded_cmd = home_dir
            elif expanded_cmd.startswith("~ "):
                expanded_cmd = home_dir + expanded_cmd[1:]
            # Replace UID/GID placeholders with shell variables
            expanded_cmd = expanded_cmd.replace("UID_PLACEHOLDER", "$MOUNT_UID")
            expanded_cmd = expanded_cmd.replace("GID_PLACEHOLDER", "$MOUNT_GID")
            lines.append(expanded_cmd)

    return "\n".join(lines)


@dataclass
class Instance:
    """Represents an EC2 instance."""

    id: str
    private_ip: str
    state: str  # running, stopped, pending
    spot: bool = False  # True if spot instance, False if on-demand
    instance_type: str = ""  # EC2 instance type (e.g., g4dn.xlarge)


class InstancePool:
    """Manages a pool of EC2 instances with stop/start lifecycle.

    Instead of terminating instances after use, this pool stops them.
    On next execution, it starts stopped instances (fast ~30s) instead
    of launching new ones (slow ~2-3min).
    """

    def __init__(
        self,
        resources: AWSResources,
        ssm: SSMSession,
    ) -> None:
        """Initialize instance pool.

        Args:
            resources: AWS resources from infrastructure.
            ssm: SSM session for command execution.
        """
        self.resources = resources
        self.ssm = ssm

    @cached_property
    def _ec2(self) -> EC2Client:
        """Lazy EC2 client."""
        import boto3

        return boto3.client("ec2", region_name=self.resources.region)

    def acquire(
        self,
        n: int,
        instance_type: str,
        ami_id: str,
        requirements_hash: str,
        apt_packages: str = "",
        python_version: str = "3.13",
        spot: SpotStrategy = Spot.Never,
        skyward_wheel_key: str | None = None,
        username: str = "ec2-user",
        volumes: tuple[Volume, ...] = (),
        pip_extra_index_url: str | None = None,
        instance_timeout: int | None = None,
    ) -> list[Instance]:
        """Acquire N instances, reusing stopped ones when possible.

        Args:
            n: Number of instances needed.
            instance_type: EC2 instance type (e.g., "g4dn.xlarge").
            ami_id: AMI ID to use for new instances.
            requirements_hash: Hash of pip/apt requirements (for matching stopped instances).
            apt_packages: Space-separated apt packages to install on boot.
            python_version: Python version to use (e.g., "3.13").
            spot: Spot strategy (Spot.Always, Spot.IfAvailable, or Spot.Never).
            skyward_wheel_key: S3 key of the skyward wheel to install.
            username: System username for the AMI (e.g., "ec2-user", "ubuntu").
                     Used for file ownership in user data script.
            volumes: Tuple of Volume objects to mount on the instances.

        Returns:
            List of running instances ready for use.
        """
        # Normalize string strategies to class instances
        spot = normalize_spot(spot)

        # Spot.Never: always use on-demand with reuse
        if isinstance(spot, _SpotNever):
            return self._acquire_on_demand(
                n, instance_type, ami_id, requirements_hash, apt_packages,
                python_version, skyward_wheel_key, username, volumes,
                pip_extra_index_url, instance_timeout,
            )

        # Spot.Always, Spot.IfAvailable, or Spot.Percent: try spot with retries
        # Note: Spot.Percent uses default retries (10) since it doesn't have retries attribute
        retries = getattr(spot, "retries", 10)
        interval = getattr(spot, "interval", 1.0)

        @retry(
            stop=stop_after_attempt(retries),
            wait=wait_fixed(interval),
            retry=retry_if_exception(_is_spot_capacity_error),
            reraise=True,
        )
        def _launch_spot() -> list[Instance]:
            launched = self._launch_instances(
                n, instance_type, ami_id, requirements_hash, apt_packages,
                python_version=python_version, spot=True,
                skyward_wheel_key=skyward_wheel_key, username=username,
                volumes=volumes, pip_extra_index_url=pip_extra_index_url,
                instance_timeout=instance_timeout,
            )
            all_ids = [i.id for i in launched]
            self._wait_running(all_ids)
            self._wait_ssm_parallel(all_ids)
            return self._get_instances(all_ids)

        try:
            return _launch_spot()
        except (RetryError, ClientError) as e:
            if isinstance(e, ClientError) and not _is_spot_capacity_error(e):
                raise  # Re-raise non-capacity errors
            # Retries exhausted
            if isinstance(spot, Spot.Always):
                raise SpotCapacityError(spot.retries, instance_type)

            # Spot.IfAvailable: fallback to on-demand
            return self._acquire_on_demand(
                n, instance_type, ami_id, requirements_hash, apt_packages,
                python_version, skyward_wheel_key, username, volumes,
                pip_extra_index_url, instance_timeout,
            )

    def _acquire_on_demand(
        self,
        n: int,
        instance_type: str,
        ami_id: str,
        requirements_hash: str,
        apt_packages: str,
        python_version: str,
        skyward_wheel_key: str | None,
        username: str,
        volumes: tuple[Volume, ...],
        pip_extra_index_url: str | None,
        instance_timeout: int | None,
    ) -> list[Instance]:
        """Acquire on-demand instances, reusing stopped ones when possible."""
        stopped = self._find_stopped_instances(instance_type, requirements_hash)

        # Start as many stopped as we can (up to n)
        to_start = stopped[:n]
        to_launch = max(0, n - len(to_start))

        started_ids: list[str] = []
        launched_ids: list[str] = []

        # Start existing instances
        if to_start:
            started_ids = self._start_instances(to_start)

        # Launch new instances for remaining
        if to_launch > 0:
            launched = self._launch_instances(
                to_launch, instance_type, ami_id, requirements_hash, apt_packages,
                python_version=python_version, spot=False,
                skyward_wheel_key=skyward_wheel_key, username=username,
                volumes=volumes, pip_extra_index_url=pip_extra_index_url,
                instance_timeout=instance_timeout,
            )
            launched_ids = [i.id for i in launched]

        # Wait for all to be ready
        all_ids = started_ids + launched_ids
        self._wait_running(all_ids)
        self._wait_ssm_parallel(all_ids)

        return self._get_instances(all_ids)

    def release(self, instances: list[Instance], spot: SpotStrategy = Spot.Never) -> None:
        """Release instances back to pool.

        - On-demand (Spot.Never): STOP (can be reused next time)
        - Spot (Spot.Always/IfAvailable): TERMINATE (cannot stop spot instances)

        Args:
            instances: Instances to release.
            spot: Spot strategy used to acquire these instances.
        """
        if not instances:
            return

        spot = normalize_spot(spot)
        instance_ids = [i.id for i in instances]

        if isinstance(spot, _SpotNever):
            self._ec2.stop_instances(InstanceIds=instance_ids)
        else:
            self._ec2.terminate_instances(InstanceIds=instance_ids)

    def terminate_all(self) -> int:
        """Terminate all skyward-managed instances (cleanup).

        Returns:
            Number of instances terminated.
        """
        response = self._ec2.describe_instances(
            Filters=[
                {"Name": f"tag:{SkywardTag.MANAGED}", "Values": ["true"]},
                {
                    "Name": "instance-state-name",
                    "Values": [
                        InstanceState.RUNNING,
                        InstanceState.STOPPED,
                        InstanceState.PENDING,
                    ],
                },
            ]
        )

        instance_ids = [
            i["InstanceId"]
            for r in response["Reservations"]
            for i in r["Instances"]
        ]

        if instance_ids:
            self._ec2.terminate_instances(InstanceIds=instance_ids)

        return len(instance_ids)

    def _find_stopped_instances(
        self, instance_type: str, requirements_hash: str
    ) -> list[Instance]:
        """Find stopped instances matching criteria."""
        response = self._ec2.describe_instances(
            Filters=[
                {"Name": f"tag:{SkywardTag.MANAGED}", "Values": ["true"]},
                {"Name": f"tag:{SkywardTag.REQUIREMENTS_HASH}", "Values": [requirements_hash]},
                {"Name": "instance-type", "Values": [instance_type]},
                {"Name": "instance-state-name", "Values": [InstanceState.STOPPED]},
            ]
        )

        return [
            Instance(
                id=i["InstanceId"],
                private_ip=i.get("PrivateIpAddress", ""),
                state="stopped",
            )
            for r in response["Reservations"]
            for i in r["Instances"]
        ]

    def _start_instances(self, instances: list[Instance]) -> list[str]:
        """Start stopped instances.

        Returns:
            List of instance IDs that were started.
        """
        if not instances:
            return []

        ids = [i.id for i in instances]
        self._ec2.start_instances(InstanceIds=ids)
        return ids

    def _launch_instances(
        self,
        n: int,
        instance_type: str,
        ami_id: str,
        requirements_hash: str,
        apt_packages: str,
        python_version: str,
        spot: bool,
        skyward_wheel_key: str | None = None,
        username: str = "ec2-user",
        volumes: tuple[Volume, ...] = (),
        pip_extra_index_url: str | None = None,
        instance_timeout: int | None = None,
    ) -> list[Instance]:
        """Launch new instances with UV setup."""
        if n == 0:
            return []

        # Generate volume mounting script (with tilde expansion for target user)
        volume_script = _generate_volume_script(volumes, username)

        user_data = self._generate_user_data(
            requirements_hash=requirements_hash,
            apt_packages=apt_packages,
            python_version=python_version,
            skyward_wheel_key=skyward_wheel_key,
            username=username,
            pip_extra_index_url=pip_extra_index_url,
            instance_timeout=instance_timeout,
            volume_script=volume_script,
        )

        run_args: dict[str, Any] = {
            "ImageId": ami_id,
            "InstanceType": instance_type,
            "MinCount": n,
            "MaxCount": n,
            "IamInstanceProfile": {"Arn": self.resources.instance_profile_arn},
            "UserData": user_data,
            "NetworkInterfaces": [
                {
                    "DeviceIndex": 0,
                    "SubnetId": self.resources.subnet_ids[0],
                    "Groups": [self.resources.security_group_id],
                    "AssociatePublicIpAddress": True,
                }
            ],
            "BlockDeviceMappings": [
                {
                    "DeviceName": "/dev/xvda",
                    "Ebs": {
                        "VolumeSize": 30,
                        "VolumeType": "gp3",
                        "DeleteOnTermination": True,
                    },
                }
            ],
            "TagSpecifications": [
                {
                    "ResourceType": "instance",
                    "Tags": [
                        {"Key": "Name", "Value": DEFAULT_INSTANCE_NAME},
                        {"Key": SkywardTag.MANAGED, "Value": "true"},
                        {"Key": SkywardTag.REQUIREMENTS_HASH, "Value": requirements_hash},
                    ],
                }
            ],
            "MetadataOptions": {"HttpTokens": "required", "HttpEndpoint": "enabled"},
        }

        if spot:
            run_args["InstanceMarketOptions"] = {
                "MarketType": "spot",
                "SpotOptions": {"SpotInstanceType": "one-time"},
            }

        if instance_timeout:
            run_args["InstanceInitiatedShutdownBehavior"] = "terminate"

        response = self._ec2.run_instances(**run_args)

        return [
            Instance(id=i["InstanceId"], private_ip="", state="pending")
            for i in response["Instances"]
        ]

    def _wait_running(self, instance_ids: list[str]) -> None:
        """Wait for all instances to be running."""
        if not instance_ids:
            return

        waiter = self._ec2.get_waiter("instance_running")
        waiter.wait(
            InstanceIds=instance_ids,
            WaiterConfig={
                "Delay": INSTANCE_RUNNING_WAIT_DELAY,
                "MaxAttempts": INSTANCE_RUNNING_MAX_ATTEMPTS,
            },
        )

    def _wait_ssm_parallel(self, instance_ids: list[str]) -> None:
        """Wait for SSM agent on all instances in parallel."""
        from skyward.conc import for_each_async

        if not instance_ids:
            return

        def wait_ssm(iid: str) -> None:
            self.ssm.wait_for_ssm_agent(iid, timeout=600)

        for_each_async(wait_ssm, instance_ids)

    def _get_instances(self, instance_ids: list[str]) -> list[Instance]:
        """Get instance details with private IPs."""
        if not instance_ids:
            return []

        @retry(
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(ClientError),
            reraise=True,
        )
        def _describe() -> DescribeInstancesResultTypeDef:
            return self._ec2.describe_instances(InstanceIds=instance_ids)

        response = _describe()

        instances = []
        for r in response["Reservations"]:
            for i in r["Instances"]:
                is_spot = i.get("InstanceLifecycle") == "spot"
                instances.append(
                    Instance(
                        id=i["InstanceId"],
                        private_ip=i["PrivateIpAddress"],
                        state=i["State"]["Name"],
                        spot=is_spot,
                        instance_type=i.get("InstanceType", ""),
                    )
                )

        instances.sort(key=lambda x: x.id)
        return instances

    def acquire_fleet(
        self,
        n: int,
        instance_types: list[str],
        ami_id: str,
        requirements_hash: str,
        apt_packages: str = "",
        python_version: str = "3.13",
        spot: SpotStrategy = Spot.IfAvailable(),
        skyward_wheel_key: str | None = None,
        username: str = "ec2-user",
        volumes: tuple[Volume, ...] = (),
        pip_extra_index_url: str | None = None,
        instance_timeout: int | None = None,
        allocation_strategy: AllocationStrategy = "price-capacity-optimized",
    ) -> list[Instance]:
        """Acquire N instances using EC2 Fleet for multi-type fallback."""
        spot = normalize_spot(spot)

        volume_script = _generate_volume_script(volumes, username)
        user_data = self._generate_user_data(
            requirements_hash, apt_packages, python_version,
            skyward_wheel_key, username, pip_extra_index_url,
            instance_timeout, volume_script,
        )

        template_id = self._create_launch_template(
            ami_id, user_data, requirements_hash, instance_timeout,
        )

        try:
            return self._launch_fleet(
                n, instance_types, self.resources.subnet_ids, template_id, spot,
                allocation_strategy,
            )
        finally:
            self._delete_launch_template(template_id)

    def _generate_user_data(
        self,
        requirements_hash: str,
        apt_packages: str,
        python_version: str,
        skyward_wheel_key: str | None,
        username: str,
        pip_extra_index_url: str | None,
        instance_timeout: int | None,
        volume_script: str,
    ) -> str:
        """Generate user data script for EC2 instances."""
        wheel_key = skyward_wheel_key or ""
        return f"""#!/bin/bash
set -ex

mkdir -p /opt/skyward
exec > /opt/skyward/bootstrap.log 2>&1
trap 'echo "Command failed: $BASH_COMMAND" > /opt/skyward/.error; echo "Exit code: $?" >> /opt/skyward/.error; echo "--- Output ---" >> /opt/skyward/.error; tail -50 /opt/skyward/bootstrap.log >> /opt/skyward/.error' ERR

# Restart SSM agent to pick up IAM role credentials
# SSM Agent v3.2+ waits 30 mins if credentials fail on first attempt
# Restarting forces immediate retry with proper IAM role
systemctl restart amazon-ssm-agent 2>/dev/null || \
systemctl restart snap.amazon-ssm-agent.amazon-ssm-agent.service 2>/dev/null || true

sleep {SSM_WAIT_SECONDS}

export SKYWARD_S3_BUCKET="{self.resources.bucket}"
export SKYWARD_REQUIREMENTS_HASH="{requirements_hash}"
export SKYWARD_APT_PACKAGES="{apt_packages}"
export SKYWARD_PYTHON_VERSION="{python_version}"
export SKYWARD_WHEEL_KEY="{wheel_key}"
export SKYWARD_USERNAME="{username}"
export SKYWARD_PIP_EXTRA_INDEX_URL="{pip_extra_index_url or ''}"
export SKYWARD_INSTANCE_TIMEOUT="{instance_timeout or ''}"

if [ -n "$SKYWARD_INSTANCE_TIMEOUT" ] && [ "$SKYWARD_INSTANCE_TIMEOUT" -gt 0 ]; then
    (sleep $SKYWARD_INSTANCE_TIMEOUT && shutdown -h now) &
fi

mkdir -p /opt/skyward
chown $SKYWARD_USERNAME:$SKYWARD_USERNAME /opt/skyward
cd /opt/skyward

if [ ! -f "/opt/skyward/uv" ]; then
    curl -LsSf {UV_INSTALL_URL} | sh
    cp ~/.local/bin/uv /opt/skyward/uv || cp /root/.local/bin/uv /opt/skyward/uv
    chmod +x /opt/skyward/uv
    chown $SKYWARD_USERNAME:$SKYWARD_USERNAME /opt/skyward/uv
fi
touch /opt/skyward/.step_uv

if [ -n "$SKYWARD_APT_PACKAGES" ]; then
    apt-get update && apt-get install -y $SKYWARD_APT_PACKAGES || \
    dnf install -y $SKYWARD_APT_PACKAGES || \
    yum install -y $SKYWARD_APT_PACKAGES || true
fi
touch /opt/skyward/.step_apt

rm -rf .venv
sudo -u $SKYWARD_USERNAME /opt/skyward/uv venv .venv --python "$SKYWARD_PYTHON_VERSION"

REQ_PATH="skyward/requirements/${{SKYWARD_REQUIREMENTS_HASH}}.txt"
aws s3 cp "s3://${{SKYWARD_S3_BUCKET}}/$REQ_PATH" requirements.txt
chown $SKYWARD_USERNAME:$SKYWARD_USERNAME requirements.txt
touch /opt/skyward/.step_download
if [ -n "$SKYWARD_PIP_EXTRA_INDEX_URL" ]; then
    sudo -u $SKYWARD_USERNAME /opt/skyward/uv pip install -r requirements.txt --extra-index-url "$SKYWARD_PIP_EXTRA_INDEX_URL" --index-strategy unsafe-best-match
else
    sudo -u $SKYWARD_USERNAME /opt/skyward/uv pip install -r requirements.txt
fi
touch /opt/skyward/.step_pip

if [ -n "$SKYWARD_WHEEL_KEY" ]; then
    WHEEL_FILE="/tmp/$(basename "$SKYWARD_WHEEL_KEY")"
    aws s3 cp "s3://${{SKYWARD_S3_BUCKET}}/skyward/${{SKYWARD_WHEEL_KEY}}" "$WHEEL_FILE"
    sudo -u $SKYWARD_USERNAME /opt/skyward/uv pip install "$WHEEL_FILE"
fi
touch /opt/skyward/.step_wheel

chown -R $SKYWARD_USERNAME:$SKYWARD_USERNAME /opt/skyward

cat > /etc/systemd/system/skyward-rpyc.service << SERVICEEOF
[Unit]
Description=Skyward RPyC Server
After=network.target

[Service]
Type=simple
User=$SKYWARD_USERNAME
WorkingDirectory=/opt/skyward
ExecStart=/opt/skyward/.venv/bin/python -m skyward.rpc
Restart=on-failure
RestartSec=2
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SERVICEEOF

systemctl daemon-reload
systemctl enable skyward-rpyc.service
systemctl start skyward-rpyc.service

for i in $(seq 1 30); do
    if systemctl is-active --quiet skyward-rpyc.service; then
        break
    fi
    sleep 0.5
done

for i in $(seq 1 60); do
    if ss -tlnp | grep -q ':18861 '; then
        break
    fi
    sleep 0.5
done
touch /opt/skyward/.step_server

{volume_script}
touch /opt/skyward/.step_volumes

touch /opt/skyward/.ready
"""

    def _create_launch_template(
        self,
        ami_id: str,
        user_data: str,
        requirements_hash: str,
        instance_timeout: int | None,
    ) -> str:
        """Create ephemeral launch template for EC2 Fleet."""
        name = f"skyward-{uuid.uuid4().hex[:8]}"

        template_data: dict[str, Any] = {
            "ImageId": ami_id,
            "IamInstanceProfile": {"Arn": self.resources.instance_profile_arn},
            "UserData": base64.b64encode(user_data.encode()).decode(),
            "MetadataOptions": {"HttpTokens": "required", "HttpEndpoint": "enabled"},
            "NetworkInterfaces": [
                {
                    "DeviceIndex": 0,
                    "Groups": [self.resources.security_group_id],
                    "AssociatePublicIpAddress": True,
                }
            ],
            "BlockDeviceMappings": [
                {
                    "DeviceName": "/dev/xvda",
                    "Ebs": {
                        "VolumeSize": 30,
                        "VolumeType": "gp3",
                        "DeleteOnTermination": True,
                    },
                }
            ],
            "TagSpecifications": [
                {
                    "ResourceType": "instance",
                    "Tags": [
                        {"Key": "Name", "Value": DEFAULT_INSTANCE_NAME},
                        {"Key": SkywardTag.MANAGED, "Value": "true"},
                        {"Key": SkywardTag.REQUIREMENTS_HASH, "Value": requirements_hash},
                    ],
                }
            ],
        }

        if instance_timeout:
            template_data["InstanceInitiatedShutdownBehavior"] = "terminate"

        response = self._ec2.create_launch_template(
            LaunchTemplateName=name,
            LaunchTemplateData=template_data,  # type: ignore[arg-type]
        )
        template_id = response["LaunchTemplate"]["LaunchTemplateId"]
        return template_id

    def _delete_launch_template(self, template_id: str) -> None:
        """Delete ephemeral launch template."""
        try:
            self._ec2.delete_launch_template(LaunchTemplateId=template_id)
        except Exception:
            pass  # Best effort cleanup

    def _launch_fleet(
        self,
        n: int,
        instance_types: list[str],
        subnet_ids: tuple[str, ...],
        template_id: str,
        spot: Spot.Always | Spot.IfAvailable | Spot.Percent | _SpotNever,
        allocation_strategy: AllocationStrategy = "price-capacity-optimized",
    ) -> list[Instance]:
        """Launch instances using EC2 Fleet API with retry."""
        retries = getattr(spot, "retries", 10)
        interval = getattr(spot, "interval", 1.0)

        if isinstance(spot, Spot.Percent):
            spot_capacity = ceil(n * spot.percentage)
            ondemand_capacity = n - spot_capacity
        elif isinstance(spot, _SpotNever):
            spot_capacity = 0
            ondemand_capacity = n
        elif isinstance(spot, Spot.Always):
            spot_capacity = n
            ondemand_capacity = 0
        else:
            spot_capacity = n
            ondemand_capacity = n

        single_subnet = subnet_ids[0]
        overrides = [
            {"InstanceType": t, "SubnetId": single_subnet, "Priority": float(i)}
            for i, t in enumerate(instance_types)
        ]

        fleet_config: dict[str, Any] = {
            "Type": "instant",
            "TargetCapacitySpecification": {
                "TotalTargetCapacity": n,
                "DefaultTargetCapacityType": "spot" if spot_capacity > 0 else "on-demand",
            },
            "LaunchTemplateConfigs": [
                {
                    "LaunchTemplateSpecification": {
                        "LaunchTemplateId": template_id,
                        "Version": "$Latest",
                    },
                    "Overrides": overrides,
                }
            ],
        }

        if spot_capacity > 0:
            fleet_config["TargetCapacitySpecification"]["SpotTargetCapacity"] = spot_capacity
            fleet_config["SpotOptions"] = {
                "AllocationStrategy": allocation_strategy,
            }

        if ondemand_capacity > 0:
            fleet_config["TargetCapacitySpecification"]["OnDemandTargetCapacity"] = ondemand_capacity

        @retry(
            stop=stop_after_attempt(retries),
            wait=wait_fixed(interval),
            retry=retry_if_exception(_is_fleet_capacity_error),
            reraise=True,
        )
        def _do_fleet() -> list[Instance]:
            response = self._ec2.create_fleet(**fleet_config)

            instance_ids: list[str] = []
            for group in response.get("Instances", []):
                instance_ids.extend(group.get("InstanceIds", []))

            got = len(instance_ids)

            if got < n:
                if instance_ids:
                    self._ec2.terminate_instances(InstanceIds=instance_ids)
                raise FleetCapacityError(n, got)

            if isinstance(spot, Spot.Percent):
                instances = self._get_instances(instance_ids)
                got_spot = sum(1 for i in instances if i.spot)
                min_spot = ceil(n * spot.percentage)

                if got_spot < min_spot:
                    self._ec2.terminate_instances(InstanceIds=instance_ids)
                    raise FleetCapacityError(n, got)

            elif isinstance(spot, Spot.Always):
                instances = self._get_instances(instance_ids)
                got_spot = sum(1 for i in instances if i.spot)
                if got_spot < n:
                    self._ec2.terminate_instances(InstanceIds=instance_ids)
                    raise FleetCapacityError(n, got_spot)

            self._wait_running(instance_ids)
            self._wait_ssm_parallel(instance_ids)

            return self._get_instances(instance_ids)

        try:
            return _do_fleet()
        except (RetryError, FleetCapacityError):
            raise SpotCapacityError(retries, ", ".join(instance_types))
