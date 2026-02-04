"""Bootstrap operations for worker isolation.

Operations for cgroups and MIG setup.
NOTE: This module is for future MIG implementation in skyward/nvidia/.
The multi-worker RPyC approach is deprecated - Ray handles workers differently.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..constants import SKYWARD_DIR

from .compose import Op
from .ops import systemd_template

if TYPE_CHECKING:
    # These types will be defined in skyward/nvidia/ when MIG is implemented
    ResourceLimits = Any
    WorkerConfig = Any


# =============================================================================
# Cgroups Operations
# =============================================================================


def cgroups(worker_count: int, limits: ResourceLimits | None) -> Op:
    """Set up cgroups v2 for worker isolation.

    Creates cgroup slices for each worker with CPU/memory limits.

    Args:
        worker_count: Number of workers to create cgroups for.
        limits: Resource limits (CPU/memory). None means no limits.

    Example:
        >>> from skyward.worker.config import ResourceLimits
        >>> limits = ResourceLimits.from_params(memory="32GB", cpu=4)
        >>> cgroups(8, limits)()
        '# Setup cgroups v2 for worker isolation...'
    """
    if limits is None:
        return lambda: f"# No cgroups limits configured\ntouch {SKYWARD_DIR}/.step_cgroups"

    cpu_max = limits.cgroup_cpu_max
    memory_max = limits.cgroup_memory_max

    def generate() -> str:
        return f"""# Setup cgroups v2 for worker isolation
CGROUP_BASE="/sys/fs/cgroup/skyward"

mkdir -p "$CGROUP_BASE"
echo "+cpu +memory" > /sys/fs/cgroup/cgroup.subtree_control 2>/dev/null || true

for i in $(seq 0 $(({worker_count} - 1))); do
    CGROUP="$CGROUP_BASE/worker-$i"
    mkdir -p "$CGROUP"
    echo "{cpu_max}" > "$CGROUP/cpu.max" 2>/dev/null || true
    echo "{memory_max}" > "$CGROUP/memory.max" 2>/dev/null || true
    echo "{memory_max}" > "$CGROUP/memory.swap.max" 2>/dev/null || true
done"""

    return generate


# =============================================================================
# Worker Environment
# =============================================================================


def worker_envs(configs: tuple[WorkerConfig, ...]) -> Op:
    """Create environment files for all workers.

    Each worker gets a file like /opt/skyward/worker-N.env with
    SKYWARD_WORKER_ID, SKYWARD_WORKER_PORT, and device env vars.

    Args:
        configs: Worker configurations.

    Example:
        >>> worker_envs(configs)()
        "cat > /opt/skyward/worker-0.env << 'EOF'..."
    """
    if not configs:
        return lambda: "# No worker env files to create"

    def generate() -> str:
        lines = []
        for config in configs:
            env_content = config.env_file_content
            lines.append(
                f"cat > {SKYWARD_DIR}/worker-{config.worker_id}.env << 'EOF'\n{env_content}EOF"
            )
        return "\n\n".join(lines)

    return generate


# =============================================================================
# MIG Setup
# =============================================================================


def mig_setup(script: str) -> Op:
    """Execute MIG partition setup script.

    The script should:
    1. Enable MIG mode on GPUs
    2. Create MIG instances
    3. Extract UUIDs and resolve placeholders in worker env files

    Args:
        script: MIG setup shell script.

    Example:
        >>> mig_setup("nvidia-smi -mig 1")()
        'nvidia-smi -mig 1'
    """
    if not script or script.isspace():
        return lambda: "# No MIG setup needed"
    return lambda: script


# =============================================================================
# Systemd Worker Services
# =============================================================================


def worker_service_unit() -> str:
    """Generate systemd unit template for workers.

    DEPRECATED: Multi-worker RPyC is replaced by Ray.
    This is kept for potential future MIG support.

    Returns the content for skyward-worker@.service.
    Uses %i for worker ID substitution.
    """
    return f"""[Unit]
Description=Skyward Worker %i
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory={SKYWARD_DIR}
EnvironmentFile={SKYWARD_DIR}/worker-%i.env
# NOTE: Command should be updated for Ray-based workers
ExecStart=/bin/echo "Worker service deprecated - use Ray"
Restart=on-failure
RestartSec=5
Slice=skyward-worker-%i.slice

# Hardening
NoNewPrivileges=yes
ProtectSystem=strict
ReadWritePaths={SKYWARD_DIR}
ReadWritePaths=/tmp

[Install]
WantedBy=multi-user.target"""


def worker_service_template(unit: str | None = None) -> Op:
    """Create systemd template for worker services.

    Args:
        unit: Custom unit content. Defaults to worker_service_unit().

    Example:
        >>> worker_service_template()()
        "cat > /etc/systemd/system/skyward-worker@.service << 'EOF'..."
    """
    content = unit or worker_service_unit()
    return systemd_template("skyward-worker", content)


def start_workers(count: int) -> Op:
    """Enable and start all worker services.

    Args:
        count: Number of workers to start.

    Example:
        >>> start_workers(8)()
        'systemctl daemon-reload...'
    """

    def generate() -> str:
        return f"""systemctl daemon-reload

for i in $(seq 0 $(({count} - 1))); do
    systemctl enable skyward-worker@$i
    systemctl start skyward-worker@$i
done"""

    return generate


def wait_for_workers(count: int, base_port: int = 18861, timeout: int = 60) -> Op:
    """Wait for all worker ports to be listening.

    Args:
        count: Number of workers.
        base_port: Base port (worker port = base_port + worker_id).
        timeout: Maximum wait time per worker in seconds.

    Example:
        >>> wait_for_workers(8)()
        'for i in $(seq 0 7); do...'
    """
    attempts = int(timeout / 0.5)

    def generate() -> str:
        return f"""for i in $(seq 0 $(({count} - 1))); do
    PORT=$(({base_port} + i))
    for attempt in $(seq 1 {attempts}); do
        if ss -tlnp 2>/dev/null | grep -q ":$PORT " || netstat -tlnp 2>/dev/null | grep -q ":$PORT "; then
            break
        fi
        sleep 0.5
    done
done"""

    return generate


# =============================================================================
# Composite Operations (DEPRECATED - use Ray)
# =============================================================================


def worker_server_ops(
    configs: tuple[WorkerConfig, ...],
    limits: ResourceLimits | None = None,
    partition_script: str = "",
) -> tuple[Op, ...]:
    """Generate all operations for multi-worker server setup.

    This is a convenience function that bundles all worker isolation
    operations into a tuple that can be spread into skyward_bootstrap().

    Args:
        configs: Worker configurations.
        limits: Resource limits for cgroups.
        partition_script: MIG/partition setup script.

    Returns:
        Tuple of operations for worker setup.

    Example:
        >>> ops = worker_server_ops(configs, limits, partition_script)
        >>> script = skyward_bootstrap(..., server_ops=ops)
    """
    from .ops import checkpoint

    ops: list[Op] = [
        cgroups(len(configs), limits),
        checkpoint(".step_cgroups"),
        worker_envs(configs),
    ]

    if partition_script:
        ops.extend(
            [
                mig_setup(partition_script),
                checkpoint(".step_partition"),
            ]
        )

    ops.extend(
        [
            worker_service_template(),
            start_workers(len(configs)),
            wait_for_workers(len(configs)),
            checkpoint(".step_server"),
        ]
    )

    return tuple(ops)
