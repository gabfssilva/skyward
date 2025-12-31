"""Bootstrap script generation for worker isolation.

Generates shell scripts for:
- cgroups v2 setup (CPU/memory limits)
- Systemd worker services (one per worker)
- Worker environment files
"""

from __future__ import annotations

from skyward.constants import SKYWARD_DIR
from skyward.worker.config import ResourceLimits, WorkerConfig


def generate_cgroups_setup(
    worker_count: int,
    limits: ResourceLimits | None,
) -> str:
    """Generate cgroups v2 setup script.

    Creates cgroup slices for each worker with CPU/memory limits.

    Args:
        worker_count: Number of workers.
        limits: Resource limits (same for all workers).

    Returns:
        Shell script fragment for bootstrap.
    """
    if limits is None:
        return f"# No cgroups limits configured\ntouch {SKYWARD_DIR}/.step_cgroups\n"

    cpu_max = limits.cgroup_cpu_max
    memory_max = limits.cgroup_memory_max

    script = f"""
# Setup cgroups v2 for worker isolation
CGROUP_BASE="/sys/fs/cgroup/skyward"

# Create base cgroup
mkdir -p "$CGROUP_BASE"

# Enable controllers
echo "+cpu +memory" > /sys/fs/cgroup/cgroup.subtree_control 2>/dev/null || true

for i in $(seq 0 $(({worker_count} - 1))); do
    CGROUP="$CGROUP_BASE/worker-$i"
    mkdir -p "$CGROUP"

    # Set CPU limit
    echo "{cpu_max}" > "$CGROUP/cpu.max" 2>/dev/null || true

    # Set memory limit
    echo "{memory_max}" > "$CGROUP/memory.max" 2>/dev/null || true

    # Enable swap limit same as memory
    echo "{memory_max}" > "$CGROUP/memory.swap.max" 2>/dev/null || true
done

touch {SKYWARD_DIR}/.step_cgroups
"""
    return script


def generate_worker_service_template() -> str:
    """Generate systemd service template for workers.

    Returns:
        Systemd unit file content.
    """
    return f"""[Unit]
Description=Skyward Worker %i
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory={SKYWARD_DIR}
EnvironmentFile={SKYWARD_DIR}/worker-%i.env
ExecStart={SKYWARD_DIR}/.venv/bin/python -m skyward.rpc
Restart=on-failure
RestartSec=5
Slice=skyward-worker-%i.slice

# Hardening
NoNewPrivileges=yes
ProtectSystem=strict
ReadWritePaths={SKYWARD_DIR}
ReadWritePaths=/tmp

[Install]
WantedBy=multi-user.target
"""


def generate_worker_env_file(config: WorkerConfig) -> str:
    """Generate environment file for a worker.

    Args:
        config: Worker configuration.

    Returns:
        Content for worker-N.env file.
    """
    return config.env_file_content


def generate_workers_setup(
    configs: tuple[WorkerConfig, ...],
    partition_setup: str = "",
) -> str:
    """Generate complete worker setup script.

    This creates:
    1. Partition setup (MIG, etc.) if needed
    2. Systemd service template
    3. Worker environment files
    4. Starts all worker services

    Args:
        configs: Worker configurations.
        partition_setup: Optional partition setup script (MIG, etc.).

    Returns:
        Shell script fragment for bootstrap.
    """
    if not configs:
        return f"# No workers to setup\ntouch {SKYWARD_DIR}/.step_workers\n"

    worker_count = len(configs)

    # Generate env file contents
    env_files_script = ""
    for config in configs:
        env_content = config.env_file_content.replace("'", "'\\''")  # Escape single quotes
        env_files_script += f"""
cat > {SKYWARD_DIR}/worker-{config.worker_id}.env << 'WORKERENVEOF'
{config.env_file_content}WORKERENVEOF
"""

    # Generate systemd service template
    service_template = generate_worker_service_template().replace("'", "'\\''")

    script = f"""
# Create worker environment files (before partition_setup so MIG can resolve placeholders)
{env_files_script}

# Partition setup (MIG, etc.) - resolves UUID placeholders in env files
{partition_setup if partition_setup else "# No partition setup needed"}
touch {SKYWARD_DIR}/.step_partition

# Create systemd service template
cat > /etc/systemd/system/skyward-worker@.service << 'SERVICEEOF'
{generate_worker_service_template()}SERVICEEOF

# Reload systemd
systemctl daemon-reload

# Enable and start worker services
for i in $(seq 0 $(({worker_count} - 1))); do
    systemctl enable skyward-worker@$i
    systemctl start skyward-worker@$i
done

# Wait for all workers to be ready
for i in $(seq 0 $(({worker_count} - 1))); do
    PORT=$((18861 + i))
    for attempt in $(seq 1 60); do
        if ss -tlnp 2>/dev/null | grep -q ":$PORT " || netstat -tlnp 2>/dev/null | grep -q ":$PORT "; then
            break
        fi
        sleep 0.5
    done
done

touch {SKYWARD_DIR}/.step_workers
"""
    return script


def generate_worker_bootstrap(
    configs: tuple[WorkerConfig, ...],
    limits: ResourceLimits | None = None,
    partition_setup: str = "",
) -> str:
    """Generate complete worker bootstrap script.

    This is the main entry point for generating worker isolation scripts.
    Combines cgroups setup and worker services into a single script.

    Args:
        configs: Worker configurations.
        limits: Resource limits for cgroups.
        partition_setup: Optional partition setup script (MIG, etc.).

    Returns:
        Complete shell script for worker bootstrap.
    """
    cgroups_script = generate_cgroups_setup(len(configs), limits)
    workers_script = generate_workers_setup(configs, partition_setup)

    return f"""
# =============================================================================
# Worker Isolation Bootstrap
# =============================================================================

{cgroups_script}

{workers_script}
"""
