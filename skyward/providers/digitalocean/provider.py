"""DigitalOcean Provider using Droplets."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, override

from skyward.events import EventCallback
from skyward.types import ComputeSpec, ExitedInstance, Instance, Provider

if TYPE_CHECKING:
    import subprocess

logger = logging.getLogger("skyward.digitalocean")


@dataclass(frozen=True)
class DigitalOcean(Provider):
    """Provider for DigitalOcean Droplets.

    Executes functions on DigitalOcean Droplets via SSH.
    Supports GPU Droplets with NVIDIA H100, H200, L40S, and more.

    SSH keys are automatically detected from ~/.ssh/id_ed25519.pub or
    ~/.ssh/id_rsa.pub and registered on DigitalOcean if needed.

    Example:
        from skyward import ComputePool, DigitalOcean, compute

        @compute
        def process(data):
            return transform(data)

        @compute
        def train(data):
            return model.fit(data)

        # Simple usage - SSH key auto-detected from ~/.ssh/
        pool = ComputePool(provider=DigitalOcean(region="nyc3"))

        # With GPU
        gpu_pool = ComputePool(
            provider=DigitalOcean(region="tor1"),
            accelerator="H100-80GB",
            pip=["torch"],
        )

        with pool:
            result = process(data) >> pool

        with gpu_pool:
            result = train(data) >> gpu_pool

    Environment Variables:
        DIGITALOCEAN_TOKEN: API token (required if not passed directly)
        DIGITALOCEAN_SSH_KEY_FINGERPRINT: SSH key fingerprint (optional)
    """

    region: str = "nyc3"
    token: str | None = None
    ssh_key_fingerprint: str | None = None
    instance_timeout: int = 300  # Auto-terminate instance after N seconds. Defaults to 5 min

    # Internal cache (not part of equality/hash)
    _resolved_fingerprint: str | None = field(
        default=None, repr=False, compare=False, hash=False
    )
    _username: str = field(default="root", repr=False, compare=False, hash=False)

    @property
    @override
    def name(self) -> str:
        """Provider name."""
        return "digitalocean"

    @property
    def api_token(self) -> str:
        """Get API token from config or environment."""
        token = self.token or os.environ.get("DIGITALOCEAN_TOKEN")
        if not token:
            raise ValueError(
                "DigitalOcean API token not provided. "
                "Set DIGITALOCEAN_TOKEN environment variable or pass token to DigitalOcean()"
            )
        return token

    @property
    def ssh_fingerprint(self) -> str:
        """Get SSH key fingerprint, auto-detecting from local keys if needed."""
        # Check explicit config first
        if self.ssh_key_fingerprint:
            return self.ssh_key_fingerprint

        # Check environment variable
        env_fingerprint = os.environ.get("DIGITALOCEAN_SSH_KEY_FINGERPRINT")
        if env_fingerprint:
            return env_fingerprint

        # Auto-detect and register
        if self._resolved_fingerprint is not None:
            return self._resolved_fingerprint

        from skyward.providers.digitalocean.ssh import get_or_create_ssh_key

        logger.info("Auto-detecting SSH key from ~/.ssh/...")
        key_info = get_or_create_ssh_key(self.api_token)

        # Cache the result (bypass frozen dataclass)
        object.__setattr__(self, "_resolved_fingerprint", key_info.fingerprint)

        return key_info.fingerprint

    @override
    def create_tunnel(self, instance: Instance) -> tuple[int, subprocess.Popen[bytes]]:
        """Create SSH tunnel to Droplet."""
        from skyward.providers.common import RPYC_PORT, create_tunnel, find_available_port

        ip = instance.get_meta("droplet_ip")
        local_port = find_available_port()
        cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            "-N",
            "-L", f"{local_port}:127.0.0.1:{RPYC_PORT}",
            f"{self._username}@{ip}",
        ]
        return create_tunnel(cmd, local_port)

    @override
    def provision(
        self,
        compute: ComputeSpec,
        on_event: EventCallback = None,
    ) -> tuple[Instance, ...]:
        """Provision DigitalOcean Droplets."""
        from skyward.providers.digitalocean.lifecycle import provision
        return provision(self, compute, on_event)

    @override
    def setup(
        self,
        instances: tuple[Instance, ...],
        compute: ComputeSpec,
        on_event: EventCallback = None,
    ) -> None:
        """Setup instances (wait for bootstrap to complete)."""
        from skyward.providers.digitalocean.lifecycle import setup
        setup(self, instances, compute, on_event)

    @override
    def shutdown(
        self,
        instances: tuple[Instance, ...],
        compute: ComputeSpec,
        on_event: EventCallback = None,
    ) -> tuple[ExitedInstance, ...]:
        """Shutdown Droplets (destroy them)."""
        from skyward.providers.digitalocean.lifecycle import shutdown
        return shutdown(self, instances, compute, on_event)

    @override
    def run_command(self, instance: Instance, command: str, timeout: int = 30) -> str:
        """Run shell command on Droplet via SSH.

        Args:
            instance: Target Droplet instance.
            command: Shell command string to execute.
            timeout: Maximum time to wait in seconds.

        Returns:
            stdout from the command.

        Raises:
            RuntimeError: If command fails (non-zero exit code).
        """
        import subprocess

        ip = instance.get_meta("droplet_ip") or instance.public_ip or instance.private_ip
        if not ip:
            raise RuntimeError(f"No IP address for instance {instance.id}")

        username = instance.get_meta("username", self._username)
        ssh_cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", f"ConnectTimeout={min(timeout, 10)}",
            f"{username}@{ip}",
            command,
        ]
        result = subprocess.run(
            ssh_cmd, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            raise RuntimeError(f"Command failed on {instance.id}: {result.stderr}")
        return result.stdout

    @override
    def discover_peers(self, cluster_id: str) -> tuple[Instance, ...]:
        """Discover peer instances in a cluster via DigitalOcean API.

        Queries DigitalOcean for all active droplets with the cluster tag.
        Droplets are sorted by private IP, with node index assigned by order.

        Args:
            cluster_id: Unique identifier for the cluster (job_id).

        Returns:
            Tuple of Instance objects for all running peers, sorted by private_ip.
        """
        from skyward.providers.digitalocean.client import get_client

        client = get_client(self.api_token)
        tag = f"skyward-cluster-{cluster_id}"

        # List droplets with the cluster tag
        response = client.droplets.list(tag_name=tag)

        instances = []
        for droplet in response.get("droplets", []):
            if droplet["status"] != "active":
                continue

            private_ip = ""
            public_ip = ""
            for network in droplet.get("networks", {}).get("v4", []):
                if network["type"] == "private":
                    private_ip = network["ip_address"]
                elif network["type"] == "public":
                    public_ip = network["ip_address"]

            instances.append(
                Instance(
                    id=str(droplet["id"]),
                    provider=self,
                    spot=False,  # DigitalOcean doesn't have spot instances
                    private_ip=private_ip or public_ip,
                    public_ip=public_ip,
                    node=0,  # Will be assigned by sort order
                    metadata=frozenset([
                        ("cluster_id", cluster_id),
                        ("droplet_id", droplet["id"]),
                        ("droplet_ip", public_ip or private_ip),
                        ("region", self.region),
                    ]),
                )
            )

        # Sort by private_ip for deterministic ordering
        instances.sort(key=lambda i: i.private_ip)

        # Reassign node index based on sort order
        return tuple(
            Instance(
                id=inst.id,
                provider=inst.provider,
                spot=inst.spot,
                private_ip=inst.private_ip,
                public_ip=inst.public_ip,
                node=idx,
                metadata=inst.metadata,
            )
            for idx, inst in enumerate(instances)
        )

