"""Common utilities shared between providers (SSH, tunnels, bootstrap)."""

from __future__ import annotations

import base64
import hashlib
import socket
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from subprocess import PIPE, Popen
from typing import TYPE_CHECKING

from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_delay,
    wait_exponential,
    wait_fixed,
)

from skyward.callback import emit
from skyward.constants import (
    DEFAULT_PYTHON,
    RPYC_PORT,
    SKYWARD_DIR,
    UV_INSTALL_URL,
)
from skyward.events import BootstrapProgress

if TYPE_CHECKING:
    from skyward.types import Instance


# =============================================================================
# Tunnel Utilities
# =============================================================================


class TunnelNotReadyError(Exception):
    """Tunnel not ready - retry."""


def find_available_port() -> int:
    """Find an available local port."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


def wait_for_tunnel(port: int, timeout: int = 30) -> None:
    """Wait for tunnel to accept connections."""

    @retry(
        stop=stop_after_delay(timeout),
        wait=wait_fixed(0.5),
        retry=retry_if_exception_type(TunnelNotReadyError),
        reraise=True,
    )
    def _check() -> None:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except (TimeoutError, ConnectionRefusedError, OSError):
            raise TunnelNotReadyError() from None

    try:
        _check()
    except RetryError as e:
        raise TimeoutError(f"Tunnel not ready on port {port}") from e


def create_tunnel(
    cmd: list[str],
    local_port: int,
    timeout: int = 30,
) -> tuple[int, Popen[bytes]]:
    """Create a tunnel process and wait for it to be ready."""
    proc = subprocess.Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)

    try:
        wait_for_tunnel(local_port, timeout=timeout)
        return local_port, proc
    except Exception:
        proc.terminate()
        raise


# =============================================================================
# Bootstrap Utilities
# =============================================================================


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """Bootstrap checkpoint for progress tracking."""

    file: str
    name: str


CHECKPOINTS: tuple[Checkpoint, ...] = (
    Checkpoint(".step_uv", "uv"),
    Checkpoint(".step_apt", "apt"),
    Checkpoint(".step_pip", "pip deps"),
    Checkpoint(".step_wheel", "skyward"),
    Checkpoint(".step_server", "server"),
)


class BootstrapNotReadyError(Exception):
    """Raised when bootstrap check should be retried."""


class BootstrapFailedError(Exception):
    """Raised when bootstrap script failed with an error."""

    def __init__(self, error_msg: str):
        self.error_msg = error_msg
        super().__init__(error_msg)


CommandRunner = Callable[[str], str]


@retry(
    stop=stop_after_delay(30),
    wait=wait_fixed(1),
    retry=retry_if_exception_type(BootstrapNotReadyError),
    reraise=True,
)
def _wait_for_rpyc_service(
    run_command: CommandRunner,
    instance_id: str,
    port: int = RPYC_PORT,
) -> None:
    """Esperar a porta RPyC estar aceitando conexões.

    Importante para AMIs onde o checkpoint .ready já existe mas
    o serviço ainda está iniciando via systemd.
    """
    check_cmd = f"nc -z 127.0.0.1 {port} && echo OK || echo WAITING"
    try:
        result = run_command(check_cmd)
        if "OK" not in result:
            raise BootstrapNotReadyError()
    except BootstrapNotReadyError:
        raise
    except Exception:
        raise BootstrapNotReadyError()


def wait_for_bootstrap(
    run_command: CommandRunner,
    instance_id: str,
    timeout: int = 300,
    extra_checkpoints: tuple[Checkpoint, ...] = (),
) -> None:
    """Wait for instance bootstrap with progress tracking."""
    all_checkpoints = CHECKPOINTS + extra_checkpoints

    checkpoint_files = " ".join(c.file for c in all_checkpoints)
    check_cmd = (
        f"cd {SKYWARD_DIR} 2>/dev/null || exit 0; "
        f"for f in {checkpoint_files} .ready .error; do "
        "[ -f $f ] && echo $f; done; "
        "if [ -f .error ]; then echo '---ERROR---'; cat .error; fi; exit 0"
    )

    completed_steps: set[str] = set()

    @retry(
        stop=stop_after_delay(timeout),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((BootstrapNotReadyError, TimeoutError)),
        reraise=True,
    )
    def _poll_bootstrap() -> None:
        nonlocal completed_steps

        try:
            stdout = run_command(check_cmd)
        except TimeoutError:
            raise
        except Exception as e:
            raise BootstrapNotReadyError() from e

        stdout = stdout.strip()

        if "---ERROR---" in stdout:
            error_msg = stdout.split("---ERROR---", 1)[1].strip()
            raise BootstrapFailedError(error_msg)

        found_files = set(stdout.split("\n")) if stdout else set()

        for checkpoint in all_checkpoints:
            if checkpoint.file in found_files and checkpoint.name not in completed_steps:
                emit(BootstrapProgress(instance_id=instance_id, step=checkpoint.name))
                completed_steps.add(checkpoint.name)

        if ".ready" in found_files:
            # Bootstrap complete - service will be started after wheel installation
            return

        raise BootstrapNotReadyError()

    try:
        _poll_bootstrap()
    except BootstrapFailedError as e:
        raise RuntimeError(f"Bootstrap failed on {instance_id}:\n{e.error_msg}") from None
    except (RetryError, BootstrapNotReadyError) as e:
        raise RuntimeError(f"Bootstrap timed out on {instance_id}") from e


def generate_base_script(
    python: str = DEFAULT_PYTHON,
    pip: tuple[str, ...] = (),
    apt: tuple[str, ...] = (),
    env: dict[str, str] | None = None,
    instance_timeout: int | None = None,
    preamble: str = "",
    postamble: str = "",
    pip_extra_index_url: str | None = None,
    worker_bootstrap: str = "",
) -> str:
    """Generate base bootstrap script for user_data."""
    pip_packages = " ".join(pip) if pip else ""
    apt_packages = " ".join(apt) if apt else ""
    env_exports = "\n".join(f'export {k}="{v}"' for k, v in (env or {}).items())
    extra_index = f"--extra-index-url {pip_extra_index_url}" if pip_extra_index_url else ""

    script = f"""#!/bin/bash
set -e

mkdir -p {SKYWARD_DIR}

exec > {SKYWARD_DIR}/bootstrap.log 2>&1

trap 'echo "Command failed: $BASH_COMMAND" > {SKYWARD_DIR}/.error; echo "Exit code: $?" >> {SKYWARD_DIR}/.error; echo "--- Output ---" >> {SKYWARD_DIR}/.error; tail -50 {SKYWARD_DIR}/bootstrap.log >> {SKYWARD_DIR}/.error' ERR

export DEBIAN_FRONTEND=noninteractive
export PATH="/root/.local/bin:$PATH"
{env_exports}

SKYWARD_INSTANCE_TIMEOUT="{instance_timeout or ''}"
if [ -n "$SKYWARD_INSTANCE_TIMEOUT" ] && [ "$SKYWARD_INSTANCE_TIMEOUT" -gt 0 ]; then
    (sleep $SKYWARD_INSTANCE_TIMEOUT && shutdown -h now) &
fi

{preamble}

if ! command -v uv &> /dev/null; then
    curl -LsSf {UV_INSTALL_URL} | sh
fi
touch {SKYWARD_DIR}/.step_uv

apt-get update -qq
apt-get install -y -qq python3 python3-venv curl ca-certificates
"""

    if apt_packages:
        script += f"apt-get install -y -qq {apt_packages}\n"

    script += f"""touch {SKYWARD_DIR}/.step_apt

cd {SKYWARD_DIR}
uv venv --python {python} venv
source venv/bin/activate

uv pip install cloudpickle rpyc
"""

    if pip_packages:
        script += f"uv pip install {pip_packages} {extra_index}\n"

    script += f"""touch {SKYWARD_DIR}/.step_pip

touch {SKYWARD_DIR}/.step_wheel
"""

    if worker_bootstrap:
        script += f"""
{postamble}

{worker_bootstrap}

touch {SKYWARD_DIR}/.step_server
touch {SKYWARD_DIR}/.ready
"""
    else:
        script += f"""
cat > /etc/systemd/system/skyward-rpyc.service << 'SERVICEEOF'
[Unit]
Description=Skyward RPyC Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory={SKYWARD_DIR}
Environment="PATH={SKYWARD_DIR}/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart={SKYWARD_DIR}/venv/bin/python -m skyward.rpc
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICEEOF

systemctl daemon-reload
systemctl enable skyward-rpyc

{postamble}

systemctl start skyward-rpyc

for i in $(seq 1 60); do
    if ss -tlnp 2>/dev/null | grep -q ':{RPYC_PORT} ' || netstat -tlnp 2>/dev/null | grep -q ':{RPYC_PORT} '; then
        break
    fi
    sleep 0.5
done
touch {SKYWARD_DIR}/.step_server

touch {SKYWARD_DIR}/.ready
"""
    return script


# =============================================================================
# SSH Utilities
# =============================================================================


SSH_KEY_PATHS = (
    "~/.ssh/id_ed25519.pub",
    "~/.ssh/id_rsa.pub",
    "~/.ssh/id_ecdsa.pub",
)


@dataclass(frozen=True)
class SSHKeyInfo:
    """Information about an SSH key."""

    id: str | None
    fingerprint: str
    public_key: str
    name: str


def find_local_ssh_key() -> tuple[Path, str] | None:
    """Find the first available SSH public key."""
    for key_path in SSH_KEY_PATHS:
        path = Path(key_path).expanduser()
        if path.exists():
            try:
                content = path.read_text().strip()
                if content:
                    return path, content
            except Exception:
                continue
    return None


def get_private_key_path() -> str | None:
    """Get path to SSH private key (without .pub extension)."""
    key_info = find_local_ssh_key()
    if key_info is None:
        return None
    pub_path, _ = key_info
    private_path = pub_path.with_suffix("")
    if private_path.exists():
        return str(private_path)
    return None


def compute_fingerprint(public_key: str) -> str:
    """Compute MD5 fingerprint of an SSH public key."""
    parts = public_key.split()
    if len(parts) < 2:
        raise ValueError(f"Invalid SSH public key format: {public_key[:50]}...")

    key_data = parts[1]
    try:
        decoded = base64.b64decode(key_data)
    except Exception as e:
        raise ValueError(f"Could not decode SSH key: {e}") from e

    md5_hash = hashlib.md5(decoded).hexdigest()
    return ":".join(md5_hash[i : i + 2] for i in range(0, len(md5_hash), 2))


def ssh_run(
    ip: str,
    username: str,
    command: str,
    timeout: int = 60,
    key_path: str | None = None,
    port: int | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run SSH command on remote instance."""
    ssh_cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-o", "ConnectTimeout=10",
    ]
    if key_path:
        ssh_cmd.extend(["-i", key_path])
    if port:
        ssh_cmd.extend(["-p", str(port)])
    ssh_cmd.extend([f"{username}@{ip}", command])
    return subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)


def scp_upload(
    ip: str,
    username: str,
    local_path: Path,
    remote_path: str,
    key_path: str | None = None,
    port: int | None = None,
) -> None:
    """Upload file to remote via SCP."""
    scp_cmd = [
        "scp",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
    ]
    if key_path:
        scp_cmd.extend(["-i", key_path])
    if port:
        scp_cmd.extend(["-P", str(port)])
    scp_cmd.extend([str(local_path), f"{username}@{ip}:{remote_path}"])
    result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"SCP failed: {result.stderr}")


def create_ssh_runner(
    ip: str,
    username: str,
    key_path: str | None = None,
) -> Callable[[str], str]:
    """Create a command runner using SSH."""

    def run_command(cmd: str) -> str:
        result = ssh_run(ip, username, cmd, timeout=30, key_path=key_path)
        if result.returncode != 0:
            raise RuntimeError(f"SSH command failed (exit {result.returncode}): {result.stderr}")
        return result.stdout

    return run_command


def build_wheel() -> Path:
    """Build skyward wheel locally."""
    skyward_dir = Path(__file__).parent.parent
    project_dir = skyward_dir.parent

    build_dir = Path("/tmp/skyward-wheel")
    build_dir.mkdir(exist_ok=True)

    result = subprocess.run(
        ["uv", "build", "--wheel", "-o", str(build_dir)],
        cwd=project_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to build wheel: {result.stderr}")

    return next(build_dir.glob("*.whl"))


def install_skyward_wheel(
    instances: tuple[Instance, ...],
    get_ip: Callable[[Instance], str],
    key_path: str | None = None,
) -> None:
    """Build and install skyward wheel on all instances (concurrent)."""
    from skyward.conc import for_each_async

    wheel_path = build_wheel()

    def install_on_instance(inst: Instance) -> None:
        ip = get_ip(inst)
        username = inst.get_meta("username", "root")

        find_uv_cmd = "which uv || find /root /home -name uv -type f 2>/dev/null | head -1"
        find_uv_result = ssh_run(ip, username, find_uv_cmd, timeout=30, key_path=key_path)
        uv_path = find_uv_result.stdout.strip()
        if not uv_path:
            raise RuntimeError(f"uv not found on {ip}")

        remote_wheel = f"{SKYWARD_DIR}/{wheel_path.name}"
        scp_upload(ip, username, wheel_path, remote_wheel, key_path=key_path)

        install_cmd = f"cd {SKYWARD_DIR} && {uv_path} add {remote_wheel}"
        result = ssh_run(ip, username, install_cmd, timeout=120, key_path=key_path)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to install wheel on {ip}: {result.stderr}")

        verify_cmd = f"{SKYWARD_DIR}/.venv/bin/python -c 'import skyward; print(skyward.__file__)'"
        verify_result = ssh_run(ip, username, verify_cmd, timeout=30, key_path=key_path)
        if verify_result.returncode != 0:
            raise RuntimeError(f"skyward not importable on {ip}: {verify_result.stderr}")

        restart_cmd = (
            "if systemctl is-active skyward-rpyc >/dev/null 2>&1; then "
            "systemctl restart skyward-rpyc; "
            "else "
            "systemctl list-units 'skyward-worker@*' --no-legend | awk '{print $1}' | xargs -r systemctl restart; "
            "fi"
        )
        ssh_run(ip, username, f"sudo bash -c \"{restart_cmd}\"", timeout=60, key_path=key_path)

    for_each_async(install_on_instance, instances)


def wait_for_ssh_bootstrap(
    instances: tuple[Instance, ...],
    get_ip: Callable[[Instance], str],
    timeout: int = 300,
    key_path: str | None = None,
) -> None:
    """Wait for bootstrap to complete on all instances via SSH (concurrent)."""
    from skyward.conc import for_each_async

    extra_checkpoints = (
        Checkpoint(".step_venv", "venv"),
        Checkpoint(".step_base", "base deps"),
    )

    username = instances[0].get_meta("username", "root")

    def wait_for_instance(inst: Instance) -> None:
        ip = get_ip(inst)
        runner = create_ssh_runner(ip, username, key_path)
        wait_for_bootstrap(
            run_command=runner,
            instance_id=inst.id,
            timeout=timeout,
            extra_checkpoints=extra_checkpoints,
        )

    for_each_async(wait_for_instance, instances)


