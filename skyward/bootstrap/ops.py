"""Core bootstrap operations.

Declarative operations for system setup: packages, files, services.
Each operation is a function returning an Op (string or callable).
"""

from __future__ import annotations

from ..constants import SKYWARD_DIR, UV_INSTALL_URL

from .compose import Op

# =============================================================================
# Package Operations
# =============================================================================


def apt(*packages: str, quiet: bool = True, update: bool = True) -> Op:
    """Install APT packages.

    Waits for dpkg lock to be released (handles unattended-upgrades).

    Args:
        *packages: Package names to install.
        quiet: Use quiet mode (-qq).
        update: Run apt-get update first.

    Example:
        >>> apt("python3", "curl")()
        'while fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do sleep 5; done\\napt-get update -qq\\napt-get install -y -qq python3 curl'
    """
    if not packages:
        return lambda: "# No APT packages to install"

    flags = "-qq" if quiet else ""
    install_flags = "-y -qq" if quiet else "-y"
    pkg_list = " ".join(packages)

    def generate() -> str:
        lines = [
            # Wait for any existing apt/dpkg processes to finish
            "while fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do sleep 5; done",
        ]
        if update:
            lines.append(f"apt-get update {flags}".strip())
        lines.append(f"apt-get install {install_flags} {pkg_list}")
        return "\n".join(lines)

    return generate


def pip(*packages: str, extra_index: str | None = None) -> Op:
    """Install pip packages using uv.

    Args:
        *packages: Package names to install.
        extra_index: Extra PyPI index URL.

    Example:
        >>> pip("torch", "transformers")()
        'uv pip install torch transformers'
    """
    if not packages:
        return lambda: "# No pip packages to install"

    pkg_list = " ".join(packages)
    extra = f" --extra-index-url {extra_index}" if extra_index else ""

    return lambda: f"uv pip install {pkg_list}{extra}"


def uv(python: str = "3.12", venv_path: str | None = None) -> Op:
    """Create virtual environment using uv.

    Args:
        python: Python version to use.
        venv_path: Path for venv. Defaults to {SKYWARD_DIR}/venv.

    Example:
        >>> uv("3.12")()
        'uv venv --python 3.12 /opt/skyward/venv'
    """
    path = venv_path or f"{SKYWARD_DIR}/venv"
    return lambda: f"uv venv --python {python} {path}"


def install_uv() -> Op:
    """Install uv package manager if not present.

    Example:
        >>> install_uv()()
        'if ! command -v uv &> /dev/null; then\\n    curl -LsSf ... | sh\\nfi'
    """
    return lambda: f"""if ! command -v uv &> /dev/null; then
    curl -LsSf {UV_INSTALL_URL} | sh
fi"""


def uv_add(*packages: str, extra_index: str | None = None) -> Op:
    """Add packages to the uv project in SKYWARD_DIR.

    Args:
        *packages: Package names to install.
        extra_index: Extra PyPI index URL.

    Example:
        >>> uv_add("torch", "transformers")()
        'cd /opt/skyward && uv add torch transformers'
    """
    if not packages:
        return lambda: "# No pip packages to install"

    pkg_list = " ".join(packages)
    extra = f" --extra-index-url {extra_index}" if extra_index else ""

    return lambda: f"cd {SKYWARD_DIR} && uv add {pkg_list}{extra}"


def uv_init(python: str = "3.12", name: str | None = None) -> Op:
    """Initialize uv project in SKYWARD_DIR.

    Creates a pyproject.toml with the specified Python version.
    Uses --no-readme to keep it minimal.

    Args:
        python: Python version to use (e.g., "3.13").
        name: Custom project name. If None, uses directory name.
              Use a custom name to avoid self-dependency issues when
              installing a package with the same name as the directory.
    """
    name_flag = f"--name {name} " if name else ""
    return lambda: f"cd {SKYWARD_DIR} && uv init {name_flag}--python {python} --no-readme"


# =============================================================================
# File Operations
# =============================================================================


def mkdir(path: str, parents: bool = True) -> Op:
    """Create directory.

    Args:
        path: Directory path to create.
        parents: Create parent directories (-p).

    Example:
        >>> mkdir("/opt/mydir")()
        'mkdir -p /opt/mydir'
    """
    flags = "-p " if parents else ""
    return lambda: f"mkdir {flags}{path}"


def file(
    path: str,
    content: str,
    mode: str | None = None,
    owner: str | None = None,
) -> Op:
    """Write content to a file using heredoc.

    Args:
        path: File path to write.
        content: File content.
        mode: Optional chmod mode (e.g., "0755").
        owner: Optional chown owner (e.g., "root:root").

    Example:
        >>> file("/etc/test.conf", "key=value")()
        "cat > /etc/test.conf << 'EOF'\\nkey=value\\nEOF"
    """

    def generate() -> str:
        lines = [f"cat > {path} << 'EOF'", content, "EOF"]
        if mode:
            lines.append(f"chmod {mode} {path}")
        if owner:
            lines.append(f"chown {owner} {path}")
        return "\n".join(lines)

    return generate


def checkpoint(name: str) -> Op:
    """Create a checkpoint file for progress tracking.

    DEPRECATED: Use phase() or phase_simple() instead for JSONL streaming.
    Checkpoint files are being replaced by real-time JSONL events.

    Args:
        name: Checkpoint name (e.g., ".step_apt", ".ready").

    Example:
        >>> checkpoint(".step_apt")()
        'touch /opt/skyward/.step_apt'
    """
    return lambda: f"touch {SKYWARD_DIR}/{name}"


def wait_for_port(port: int, timeout: int = 60, interval: float = 0.5) -> Op:
    """Wait for a port to be listening.

    Args:
        port: Port number to check.
        timeout: Maximum wait time in seconds.
        interval: Check interval in seconds.

    Example:
        >>> wait_for_port(8080, timeout=30)()
        'for i in $(seq 1 60); do...'
    """
    attempts = int(timeout / interval)

    def generate() -> str:
        return f"""for i in $(seq 1 {attempts}); do
    if ss -tlnp 2>/dev/null | grep -q ':{port} ' || netstat -tlnp 2>/dev/null | grep -q ':{port} '; then
        break
    fi
    sleep {interval}
done"""

    return generate


# =============================================================================
# Environment Operations
# =============================================================================


def env_export(**variables: str) -> Op:
    """Export environment variables.

    Args:
        **variables: Variable name=value pairs.

    Example:
        >>> env_export(HOME="/root", PATH="/usr/bin")()
        'export HOME="/root"\\nexport PATH="/usr/bin"'
    """
    if not variables:
        return lambda: "# No environment variables"

    def generate() -> str:
        return "\n".join(f'export {k}="{v}"' for k, v in variables.items())

    return generate


def shell_vars(**variables: str) -> Op:
    """Define shell variables by executing commands remotely.

    Each variable is set by executing its shell command and capturing stdout.
    Variables are exported for use in subsequent commands (pip, apt, env, etc.).

    Commands run with `set -e`, so any failure aborts the bootstrap.

    Args:
        **variables: Variable name -> shell command pairs.

    Example:
        >>> shell_vars(CUDA_VER="nvidia-smi ... | head -1")()
        '# Resolve shell variables\\nCUDA_VER=$(nvidia-smi ...)\\nexport CUDA_VER'

    Usage in Image:
        Image(
            pip=["torch==${CUDA_VER}"],
            shell_vars={"CUDA_VER": "nvidia-smi ... | head -1"},
        )
    """
    if not variables:
        return lambda: "# No shell variables to resolve"

    def generate() -> str:
        lines = ["# Resolve shell variables"]
        for name, cmd in variables.items():
            lines.append(f"{name}=$({cmd})")
            lines.append(f"export {name}")
        return "\n".join(lines)

    return generate


def instance_timeout(seconds: int, shutdown_command: str = "shutdown -h now") -> Op:
    """Set up automatic instance shutdown after timeout.

    Args:
        seconds: Timeout in seconds (0 to disable).
        shutdown_command: Shell command to execute on timeout.

    Example:
        >>> instance_timeout(3600)()
        'SKYWARD_INSTANCE_TIMEOUT="3600"...'
    """

    def generate() -> str:
        return f"""SKYWARD_INSTANCE_TIMEOUT="{seconds}"
if [ -n "$SKYWARD_INSTANCE_TIMEOUT" ] && [ "$SKYWARD_INSTANCE_TIMEOUT" -gt 0 ]; then
    cat > {SKYWARD_DIR}/auto-shutdown.sh << 'SHUTDOWN'
{shutdown_command}
SHUTDOWN
    chmod +x {SKYWARD_DIR}/auto-shutdown.sh
    nohup bash -c "sleep $SKYWARD_INSTANCE_TIMEOUT && {SKYWARD_DIR}/auto-shutdown.sh" &
fi"""

    return generate


# =============================================================================
# Shell Operations
# =============================================================================


def shell(cmd: str) -> Op:
    """Execute a raw shell command.

    This is a pass-through for arbitrary shell commands.
    Prefer using specific operations when available.

    Args:
        cmd: Shell command to execute.

    Example:
        >>> shell("echo hello")()
        'echo hello'
    """
    return lambda: cmd


def cd(path: str) -> Op:
    """Change directory.

    Args:
        path: Directory path.

    Example:
        >>> cd("/opt/skyward")()
        'cd /opt/skyward'
    """
    return lambda: f"cd {path}"


# =============================================================================
# AWS Operations
# =============================================================================


def grid_driver() -> Op:
    """Install NVIDIA GRID driver for fractional GPU instances (G6f/Gr6f).

    Downloads and installs the GRID driver from AWS S3. Required for
    instances with fractional GPU (vGPU) which don't support Tesla drivers.

    Example:
        >>> grid_driver()()
        '# Install NVIDIA GRID driver...'
    """
    return lambda: """# Install NVIDIA GRID driver for fractional GPU
apt-get update -qq
apt-get install -y -qq gcc-12 make linux-modules-extra-$(uname -r) awscli
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100
ln -sf /usr/bin/gcc /usr/bin/cc

aws s3 cp s3://ec2-linux-nvidia-drivers/latest/NVIDIA-Linux-x86_64-580.105.08-grid-aws.run /tmp/nvidia-grid.run
chmod +x /tmp/nvidia-grid.run
/tmp/nvidia-grid.run --silent --dkms
rm -f /tmp/nvidia-grid.run"""


# =============================================================================
# Phase Operations (for JSONL streaming)
# =============================================================================


def phase(name: str, *ops: Op) -> Op:
    """Wrap operations in a named phase with JSONL streaming.

    Emits phase started/completed events and streams command output line by line.
    Each line of output becomes a console event in the JSONL log.

    Args:
        name: Phase name (e.g., "apt", "pip", "uv").
        *ops: Operations to run in this phase.

    Example:
        >>> phase("apt", apt("python3", "curl"))()
        'emit_phase "started" "apt"\\n...\\nemit_phase "completed" "apt" "$elapsed"'
    """
    from .compose import resolve

    def generate() -> str:
        inner = "\n".join(resolve(op) for op in ops if op is not None)
        # Use run_phase for streaming output
        # Escape single quotes in the inner script
        escaped = inner.replace("'", "'\"'\"'")
        return f"run_phase \"{name}\" bash -c '{escaped}'"

    return generate


def phase_simple(name: str, *ops: Op) -> Op:
    """Wrap operations in a phase without output streaming.

    Use this for fast operations where line-by-line output streaming
    is not needed. Still emits phase started/completed events.

    Args:
        name: Phase name.
        *ops: Operations to run.
    """
    from .compose import resolve

    def generate() -> str:
        inner = "\n".join(resolve(op) for op in ops if op is not None)
        return f"""emit_phase "started" "{name}"
_phase_start=$(date +%s%N)
{inner}
_phase_elapsed=$(awk "BEGIN {{printf \\"%.2f\\", ($(date +%s%N) - $_phase_start) / 1000000000}}")
emit_phase "completed" "{name}" "$_phase_elapsed\""""

    return generate


def emit_bootstrap_complete() -> Op:
    """Emit the final bootstrap completed event.

    Call this at the end of the bootstrap script to signal completion.
    """
    return lambda: 'emit_phase "completed" "bootstrap"'


def start_metrics() -> Op:
    """Start the metrics daemon after bootstrap.

    The daemon emits metrics events to events.jsonl every 200ms.
    GPU metrics are cached and only updated every ~1s (5 iterations).

    Should be called after emit_bootstrap_complete().
    """
    return lambda: "start_metrics_daemon"


def stop_metrics() -> Op:
    """Stop the metrics daemon.

    Kills the background process that emits metrics.
    """
    return lambda: "stop_metrics_daemon"
