"""Core bootstrap operations.

Declarative operations for system setup: packages, files, services.
Each operation is a function returning an Op (string or callable).
"""

from __future__ import annotations

from skyward.constants import SKYWARD_DIR, UV_INSTALL_URL

from .compose import Op

# =============================================================================
# Package Operations
# =============================================================================


def apt(*packages: str, quiet: bool = True, update: bool = True) -> Op:
    """Install APT packages.

    Args:
        *packages: Package names to install.
        quiet: Use quiet mode (-qq).
        update: Run apt-get update first.

    Example:
        >>> apt("python3", "curl")()
        'apt-get update -qq\\napt-get install -y -qq python3 curl'
    """
    if not packages:
        return lambda: "# No APT packages to install"

    flags = "-qq" if quiet else ""
    install_flags = "-y -qq" if quiet else "-y"
    pkg_list = " ".join(packages)

    def generate() -> str:
        lines = []
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


def activate(venv_path: str | None = None) -> Op:
    """Activate virtual environment.

    Args:
        venv_path: Path to venv. Defaults to {SKYWARD_DIR}/venv.

    Example:
        >>> activate()()
        'source /opt/skyward/venv/bin/activate'
    """
    path = venv_path or f"{SKYWARD_DIR}/venv"
    return lambda: f"source {path}/bin/activate"


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


def uv_init(python: str = "3.12") -> Op:
    """Initialize uv project in SKYWARD_DIR.

    Creates a pyproject.toml with the specified Python version.
    Uses --no-readme --no-pin-python to keep it minimal.

    Args:
        python: Python version to use (e.g., "3.13").
    """
    return lambda: f"cd {SKYWARD_DIR} && uv init --python {python} --no-readme"

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

    Args:
        name: Checkpoint name (e.g., ".step_apt", ".ready").

    Example:
        >>> checkpoint(".step_apt")()
        'touch /opt/skyward/.step_apt'
    """
    return lambda: f"touch {SKYWARD_DIR}/{name}"


# =============================================================================
# Service Operations
# =============================================================================


def systemd(
    name: str,
    unit: str,
    enable: bool = True,
    start: bool = True,
    daemon_reload: bool = True,
) -> Op:
    """Create and manage a systemd service.

    Args:
        name: Service name (without .service).
        unit: Unit file content.
        enable: Enable the service.
        start: Start the service.
        daemon_reload: Reload systemd daemon.

    Example:
        >>> systemd("myservice", "[Unit]\\nDescription=My Service")()
        "cat > /etc/systemd/system/myservice.service << 'EOF'..."
    """

    def generate() -> str:
        lines = [file(f"/etc/systemd/system/{name}.service", unit)()]
        if daemon_reload:
            lines.append("systemctl daemon-reload")
        if enable:
            lines.append(f"systemctl enable {name}")
        if start:
            lines.append(f"systemctl start {name}")
        return "\n".join(lines)

    return generate


def systemd_template(name: str, unit: str) -> Op:
    """Create a systemd template unit (name@.service).

    Args:
        name: Service name (e.g., "myservice" -> "myservice@.service").
        unit: Unit file content with %i placeholders.

    Example:
        >>> systemd_template("worker", "[Service]\\nEnvironmentFile=worker-%i.env")()
        "cat > /etc/systemd/system/worker@.service << 'EOF'..."
    """
    return file(f"/etc/systemd/system/{name}@.service", unit)


def nohup_service(
    name: str,
    command: str,
    working_dir: str = SKYWARD_DIR,
    env: dict[str, str] | None = None,
    log_file: str | None = None,
) -> Op:
    """Start a service in background using nohup (for Docker containers without systemd).

    Args:
        name: Service name (for identification).
        command: Command to run.
        working_dir: Working directory.
        env: Environment variables.
        log_file: Path to log file (default: /var/log/{name}.log).

    Example:
        >>> nohup_service("myservice", "python -m myapp", env={"FOO": "bar"})()
        '# Start myservice in background\\ncd /opt/skyward\\nFOO="bar" nohup python...'
    """

    def generate() -> str:
        log = log_file or f"/var/log/{name}.log"
        env_exports = ""
        if env:
            env_exports = " ".join(f'{k}="{v}"' for k, v in env.items()) + " "

        return f"""# Start {name} in background
cd {working_dir}
{env_exports}nohup {command} > {log} 2>&1 &
echo $! > /var/run/{name}.pid"""

    return generate


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
# SSH Operations
# =============================================================================


def inject_ssh_key(public_key: str) -> Op:
    """Inject SSH public key into authorized_keys for root and ubuntu users.

    Args:
        public_key: SSH public key content (e.g., "ssh-ed25519 AAAA... user@host").

    Example:
        >>> inject_ssh_key("ssh-ed25519 AAAA...")()
        'mkdir -p /root/.ssh && echo "ssh-ed25519 AAAA..." >> /root/.ssh/authorized_keys...'
    """
    # Escape any quotes in the key
    escaped_key = public_key.replace('"', '\\"')

    def generate() -> str:
        return f"""# Inject SSH public key
for user_home in /root /home/ubuntu /home/ec2-user; do
    if [ -d "$user_home" ]; then
        mkdir -p "$user_home/.ssh"
        chmod 700 "$user_home/.ssh"
        echo "{escaped_key}" >> "$user_home/.ssh/authorized_keys"
        chmod 600 "$user_home/.ssh/authorized_keys"
        chown -R $(stat -c '%U:%G' "$user_home") "$user_home/.ssh" 2>/dev/null || true
    fi
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


def instance_timeout(seconds: int) -> Op:
    """Set up automatic instance shutdown after timeout.

    Args:
        seconds: Timeout in seconds (0 to disable).

    Example:
        >>> instance_timeout(3600)()
        'SKYWARD_INSTANCE_TIMEOUT="3600"...'
    """

    def generate() -> str:
        return f"""SKYWARD_INSTANCE_TIMEOUT="{seconds}"
if [ -n "$SKYWARD_INSTANCE_TIMEOUT" ] && [ "$SKYWARD_INSTANCE_TIMEOUT" -gt 0 ]; then
    (sleep $SKYWARD_INSTANCE_TIMEOUT && shutdown -h now) &
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


def s3_pip_install(
    bucket: str,
    requirements_hash: str,
    extra_index: str | None = None,
) -> Op:
    """Download and install pip requirements from S3.

    Args:
        bucket: S3 bucket name.
        requirements_hash: Hash identifying the requirements file.
        extra_index: Extra PyPI index URL.

    Example:
        >>> s3_pip_install("mybucket", "abc123")()
        'aws s3 cp "s3://mybucket/skyward/requirements/abc123.txt"...'
    """
    extra = f' --extra-index-url "{extra_index}" --index-strategy unsafe-best-match' if extra_index else ""

    return lambda: f"""aws s3 cp "s3://{bucket}/skyward/requirements/{requirements_hash}.txt" {SKYWARD_DIR}/requirements.txt
uv pip install -r {SKYWARD_DIR}/requirements.txt{extra}"""


def s3_wheel(bucket: str, wheel_key: str) -> Op:
    """Download and install skyward wheel from S3.

    Args:
        bucket: S3 bucket name.
        wheel_key: S3 key for the wheel file.

    Example:
        >>> s3_wheel("mybucket", "wheel/abc123/skyward-0.1.0.whl")()
        'aws s3 cp "s3://mybucket/skyward/wheel/..."...'
    """
    return lambda: f"""WHEEL_FILE="/tmp/$(basename "{wheel_key}")"
aws s3 cp "s3://{bucket}/skyward/{wheel_key}" "$WHEEL_FILE"
cd {SKYWARD_DIR} && uv add "$WHEEL_FILE\""""
