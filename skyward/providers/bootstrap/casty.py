"""Bootstrap operations for Casty actor system setup.

Operations for installing Casty and starting head/worker nodes.
Each node runs a ClusteredActorSystem with discoverable worker actors
for direct communication via ClusterClient.
"""

from __future__ import annotations

from skyward.providers.bootstrap.compose import SKYWARD_DIR

VENV_DIR = f"{SKYWARD_DIR}/.venv"
from .compose import Op  # noqa: E402
from .ops import pip, wait_for_port  # noqa: E402

CASTY_PORT = 25520


def _installed_casty_version() -> str:
    from importlib.metadata import version
    return version("casty")


def casty_install(version: str | None = None) -> Op:
    ver = version or _installed_casty_version()
    return pip(f"casty=={ver}")


def casty_service(
    node_id: int,
    head_ip: str | None,
    port: int = CASTY_PORT,
) -> Op:
    python_bin = f"{VENV_DIR}/bin/python"

    def generate() -> str:
        seeds_arg = ""
        if node_id > 0:
            if head_ip is None:
                raise ValueError("worker nodes require head_ip")
            seeds_arg = f"--seeds {head_ip}:{port}"

        cmd = (
            f"{python_bin} -m skyward.infra.worker "
            f"--node-id {node_id} "
            f"--port {port} "
            f"{seeds_arg}"
        )

        return f"""nohup {cmd} > /var/log/casty.log 2>&1 &
echo $! > /var/run/casty.pid

# Stream Casty logs to events.jsonl in background
(tail -f /var/log/casty.log 2>/dev/null | while IFS= read -r line; do
    emit_console "[casty] $line"
done) &"""

    return generate


def server_ops(
    node_id: int,
    head_ip: str | None,
    casty_version: str | None = None,
) -> tuple[Op, ...]:
    ops: list[Op] = [
        casty_install(casty_version),
        casty_service(
            node_id=node_id,
            head_ip=head_ip,
        ),
        wait_for_port(CASTY_PORT, timeout=60),
    ]

    return tuple(ops)
