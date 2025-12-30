"""Worker configuration types.

Defines resource limits and worker configuration for isolated execution.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from skyward.constants import RPYC_PORT


# Memory size units
_SIZE_UNITS: Final[dict[str, int]] = {
    "B": 1,
    "K": 1024,
    "KB": 1024,
    "M": 1024 ** 2,
    "MB": 1024 ** 2,
    "G": 1024 ** 3,
    "GB": 1024 ** 3,
    "T": 1024 ** 4,
    "TB": 1024 ** 4,
}

_SIZE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*([KMGT]B?|B)?\s*$",
    re.IGNORECASE,
)


def parse_memory_size(size: str | int | None) -> int | None:
    """Parse memory size string to bytes.

    Args:
        size: Memory size as string (e.g., "32GB", "512MB") or int (bytes).

    Returns:
        Size in bytes, or None if size is None.

    Raises:
        ValueError: If size format is invalid.

    Examples:
        >>> parse_memory_size("32GB")
        34359738368
        >>> parse_memory_size("512MB")
        536870912
        >>> parse_memory_size(1024)
        1024
    """
    if size is None:
        return None

    if isinstance(size, int):
        return size

    match = _SIZE_PATTERN.match(size)
    if not match:
        raise ValueError(
            f"Invalid memory size: {size!r}. "
            "Expected format: NUMBER[UNIT] (e.g., '32GB', '512MB', '1024')"
        )

    value = float(match.group(1))
    unit = (match.group(2) or "B").upper()

    if unit not in _SIZE_UNITS:
        raise ValueError(f"Unknown unit: {unit}")

    return int(value * _SIZE_UNITS[unit])


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    """Resource limits for a worker via cgroups v2.

    Attributes:
        memory_bytes: Memory limit in bytes (None = unlimited)
        cpu_cores: CPU core limit (None = unlimited)
        cpu_period_us: CPU period in microseconds (default 100ms)
    """

    memory_bytes: int | None = None
    cpu_cores: int | None = None
    cpu_period_us: int = 100_000  # 100ms default

    @classmethod
    def from_params(
        cls,
        memory: str | int | None = None,
        cpu: int | None = None,
    ) -> ResourceLimits | None:
        """Create ResourceLimits from user-friendly parameters.

        Args:
            memory: Memory limit (e.g., "32GB", "512MB", or bytes).
            cpu: CPU core limit.

        Returns:
            ResourceLimits if any limits specified, None otherwise.
        """
        memory_bytes = parse_memory_size(memory)

        if memory_bytes is None and cpu is None:
            return None

        return cls(memory_bytes=memory_bytes, cpu_cores=cpu)

    @property
    def cpu_quota_us(self) -> int | None:
        """CPU quota in microseconds for cgroups cpu.max.

        Returns cpu_cores * cpu_period_us, or None if no limit.
        """
        if self.cpu_cores is None:
            return None
        return self.cpu_cores * self.cpu_period_us

    @property
    def cgroup_cpu_max(self) -> str:
        """Value for cgroups cpu.max file.

        Returns "$quota $period" or "max $period" if unlimited.
        """
        if self.cpu_quota_us is None:
            return f"max {self.cpu_period_us}"
        return f"{self.cpu_quota_us} {self.cpu_period_us}"

    @property
    def cgroup_memory_max(self) -> str:
        """Value for cgroups memory.max file.

        Returns bytes as string, or "max" if unlimited.
        """
        if self.memory_bytes is None:
            return "max"
        return str(self.memory_bytes)


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    """Configuration for a single worker process.

    Attributes:
        worker_id: Unique worker ID (0-indexed)
        port: RPyC server port for this worker
        device_env: Environment variables for device assignment
        limits: Resource limits (cgroups)
    """

    worker_id: int
    port: int
    device_env: frozenset[tuple[str, str]]
    limits: ResourceLimits | None = None

    @classmethod
    def create(
        cls,
        worker_id: int,
        device_env: dict[str, str] | None = None,
        limits: ResourceLimits | None = None,
        base_port: int = RPYC_PORT,
    ) -> WorkerConfig:
        """Create a WorkerConfig.

        Args:
            worker_id: Worker ID (0-indexed).
            device_env: Device environment variables (e.g., CUDA_VISIBLE_DEVICES).
            limits: Resource limits.
            base_port: Base port for RPyC (worker port = base_port + worker_id).

        Returns:
            WorkerConfig instance.
        """
        return cls(
            worker_id=worker_id,
            port=base_port + worker_id,
            device_env=frozenset((device_env or {}).items()),
            limits=limits,
        )

    @property
    def env_dict(self) -> dict[str, str]:
        """Device environment as mutable dict."""
        return dict(self.device_env)

    @property
    def env_file_content(self) -> str:
        """Generate content for worker-N.env file.

        Returns:
            Environment file content with SKYWARD_WORKER_* vars and device env.
        """
        lines = [
            f"SKYWARD_WORKER_ID={self.worker_id}",
            f"SKYWARD_WORKER_PORT={self.port}",
        ]
        for key, value in sorted(self.device_env):
            lines.append(f"{key}={value}")
        return "\n".join(lines) + "\n"

    @property
    def cgroup_slice_name(self) -> str:
        """Systemd slice name for this worker."""
        return f"skyward-worker-{self.worker_id}.slice"


def generate_worker_configs(
    worker_count: int,
    device_env_fn: callable[[int], dict[str, str]],
    limits: ResourceLimits | None = None,
    base_port: int = RPYC_PORT,
) -> tuple[WorkerConfig, ...]:
    """Generate configurations for all workers.

    Args:
        worker_count: Number of workers.
        device_env_fn: Function (worker_id) -> device env vars.
        limits: Resource limits (same for all workers).
        base_port: Base port for RPyC.

    Returns:
        Tuple of WorkerConfig for each worker.
    """
    return tuple(
        WorkerConfig.create(
            worker_id=i,
            device_env=device_env_fn(i),
            limits=limits,
            base_port=base_port,
        )
        for i in range(worker_count)
    )
