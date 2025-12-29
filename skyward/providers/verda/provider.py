"""Verda Provider for GPU instances."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, override

from skyward.types import ComputeSpec, ExitedInstance, Instance, Provider

if TYPE_CHECKING:
    import subprocess


@dataclass(frozen=True)
class Verda(Provider):
    """Provider for Verda Cloud (formerly DataCrunch) GPU instances.

    Executes functions on Verda GPU instances via SSH.
    Supports NVIDIA GPUs: V100, A100, H100, H200, L40S, GB200.

    SSH keys are automatically detected from ~/.ssh/id_ed25519.pub or
    ~/.ssh/id_rsa.pub and registered on Verda if needed.

    Example:
        from skyward import ComputePool, Verda, compute

        @compute
        def train(data):
            return model.fit(data)

        # Simple usage - credentials from environment
        pool = ComputePool(
            provider=Verda(region="FIN-01"),
            accelerator="H100-80GB",
            pip=["torch"],
        )

        with pool:
            result = train(data) >> pool

    Environment Variables:
        VERDA_CLIENT_ID: API client ID (required if not passed directly)
        VERDA_CLIENT_SECRET: API client secret (required if not passed directly)
    """

    region: str = "FIN-01"  # Finland datacenter
    client_id: str | None = None
    client_secret: str | None = None
    ssh_key_id: str | None = None
    instance_timeout: int = 300  # Auto-terminate instance after N seconds. Defaults to 5 min

    # Internal cache (not part of equality/hash)
    _resolved_ssh_key_id: str | None = field(
        default=None, repr=False, compare=False, hash=False
    )
    _username: str = field(default="root", repr=False, compare=False, hash=False)
    _instances: list = field(default_factory=list, repr=False, compare=False, hash=False)
    _cluster_id: str | None = field(default=None, repr=False, compare=False, hash=False)

    @property
    @override
    def name(self) -> str:
        """Provider name."""
        return "verda"

    @property
    def client_id_resolved(self) -> str:
        """Get client ID from config or environment."""
        client_id = self.client_id or os.environ.get("VERDA_CLIENT_ID")
        if not client_id:
            raise ValueError(
                "Verda client ID not provided. "
                "Set VERDA_CLIENT_ID environment variable or pass client_id to Verda()"
            )
        return client_id

    @property
    def client_secret_resolved(self) -> str:
        """Get client secret from config or environment."""
        client_secret = self.client_secret or os.environ.get("VERDA_CLIENT_SECRET")
        if not client_secret:
            raise ValueError(
                "Verda client secret not provided. "
                "Set VERDA_CLIENT_SECRET environment variable or pass client_secret to Verda()"
            )
        return client_secret

    @property
    def ssh_key_resolved(self) -> str:
        """Get SSH key ID, auto-detecting from local keys if needed."""
        # Check explicit config first
        if self.ssh_key_id:
            return self.ssh_key_id

        # Check cached value
        if self._resolved_ssh_key_id is not None:
            return self._resolved_ssh_key_id

        # Auto-detect and register
        from skyward.providers.verda.client import get_client
        from skyward.providers.verda.ssh import get_or_create_ssh_key

        client = get_client(self.client_id_resolved, self.client_secret_resolved)
        key_info = get_or_create_ssh_key(client)

        # Cache the result (bypass frozen dataclass)
        object.__setattr__(self, "_resolved_ssh_key_id", key_info.id)

        return key_info.id

    @override
    def create_tunnel(self, instance: Instance) -> tuple[int, subprocess.Popen[bytes]]:
        """Create SSH tunnel to instance."""
        from skyward.providers.common import RPYC_PORT, create_tunnel, find_available_port
        from skyward.providers.verda.ssh import find_local_ssh_key

        ip = instance.get_meta("instance_ip") or instance.public_ip
        local_port = find_available_port()

        # Get SSH private key path
        key_info = find_local_ssh_key()
        key_path = str(key_info[0].with_suffix("")) if key_info else None

        cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            "-N",
        ]
        if key_path:
            cmd.extend(["-i", key_path])
        cmd.extend(["-L", f"{local_port}:127.0.0.1:{RPYC_PORT}", f"{self._username}@{ip}"])
        return create_tunnel(cmd, local_port)

    @override
    def provision(
        self,
        compute: ComputeSpec,
    ) -> tuple[Instance, ...]:
        """Provision Verda GPU instances."""
        from skyward.providers.verda.lifecycle import provision
        return provision(self, compute)

    @override
    def setup(
        self,
        instances: tuple[Instance, ...],
        compute: ComputeSpec,
    ) -> None:
        """Setup instances (wait for bootstrap to complete)."""
        from skyward.providers.verda.lifecycle import setup
        setup(self, instances, compute)

    @override
    def shutdown(
        self,
        instances: tuple[Instance, ...],
        compute: ComputeSpec,
    ) -> tuple[ExitedInstance, ...]:
        """Shutdown instances (delete them)."""
        from skyward.providers.verda.lifecycle import shutdown
        return shutdown(self, instances, compute)

    @override
    def run_command(self, instance: Instance, command: str, timeout: int = 30) -> str:
        """Run shell command on instance via SSH.

        Args:
            instance: Target instance.
            command: Shell command string to execute.
            timeout: Maximum time to wait in seconds.

        Returns:
            stdout from the command.

        Raises:
            RuntimeError: If command fails (non-zero exit code).
        """
        import subprocess

        from skyward.providers.verda.ssh import find_local_ssh_key

        ip = instance.get_meta("instance_ip") or instance.public_ip or instance.private_ip
        if not ip:
            raise RuntimeError(f"No IP address for instance {instance.id}")

        # Get SSH private key path
        key_info = find_local_ssh_key()
        key_path = str(key_info[0].with_suffix("")) if key_info else None

        username = instance.get_meta("username", self._username)
        ssh_cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", f"ConnectTimeout={min(timeout, 10)}",
        ]
        if key_path:
            ssh_cmd.extend(["-i", key_path])
        ssh_cmd.extend([f"{username}@{ip}", command])
        result = subprocess.run(
            ssh_cmd, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            raise RuntimeError(f"Command failed on {instance.id}: {result.stderr}")
        return result.stdout

    @override
    def discover_peers(self, cluster_id: str) -> tuple[Instance, ...]:
        """Discover peer instances in a cluster via Verda API.

        Queries Verda for all running instances with the cluster hostname pattern.
        Instances are sorted by IP, with node index assigned by order.

        Args:
            cluster_id: Unique identifier for the cluster (job_id).

        Returns:
            Tuple of Instance objects for all running peers, sorted by IP.
        """
        from skyward.providers.verda.client import get_client

        client = get_client(self.client_id_resolved, self.client_secret_resolved)

        # List all instances and filter by hostname pattern
        hostname_pattern = f"skyward-{cluster_id}-"
        all_instances = client.instances.get()

        instances = []
        for vinst in all_instances:
            # Check hostname matches our cluster
            if not hasattr(vinst, "hostname") or not vinst.hostname.startswith(hostname_pattern):
                continue

            # Check status is running
            if vinst.status != "running":
                continue

            ip = getattr(vinst, "ip", "") or ""

            instances.append(
                Instance(
                    id=str(vinst.id),
                    provider=self,
                    spot=getattr(vinst, "is_spot", False),
                    private_ip=ip,  # Verda typically uses public IPs
                    public_ip=ip,
                    node=0,  # Will be assigned by sort order
                    metadata=frozenset([
                        ("cluster_id", cluster_id),
                        ("instance_id", vinst.id),
                        ("instance_ip", ip),
                        ("username", self._username),
                    ]),
                )
            )

        # Sort by IP for deterministic ordering
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
