"""Bootstrap operations for Casty actor system setup.

Operations for installing Casty and starting head/worker nodes.
Each node runs a ClusteredActorSystem with a Starlette HTTP API for job submission.
"""

from __future__ import annotations

from ..constants import VENV_DIR
from .compose import Op
from .ops import pip, shell, wait_for_port

CASTY_PORT = 25520
HTTP_PORT = 8265


def casty_install(version: str = "0.3.1") -> Op:
    return pip(f"casty=={version}", "starlette", "uvicorn")


def casty_service(
    node_id: int,
    head_ip: str | None,
    port: int = CASTY_PORT,
    http_port: int = HTTP_PORT,
) -> Op:
    python_bin = f"{VENV_DIR}/bin/python"

    def generate() -> str:
        seeds_arg = ""
        if node_id > 0:
            if head_ip is None:
                raise ValueError("worker nodes require head_ip")
            seeds_arg = f"--seeds {head_ip}:{port}"

        cmd = (
            f"{python_bin} -m skyward.casty_worker "
            f"--node-id {node_id} "
            f"--port {port} "
            f"--http-port {http_port} "
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
    casty_version: str = "0.3.1",
) -> tuple[Op, ...]:
    ops: list[Op] = [
        casty_install(casty_version),
        casty_service(
            node_id=node_id,
            head_ip=head_ip,
        ),
    ]

    if node_id == 0:
        ops.append(wait_for_port(HTTP_PORT, timeout=60))
    else:
        ops.append(shell("sleep 5"))

    return tuple(ops)


__all__ = [
    "casty_install",
    "casty_service",
    "server_ops",
]
