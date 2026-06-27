"""On-disk per-session handles for server reattach after a restart.

Persist provider config + cluster + per-node SSH coordinates so the server
can re-adopt live instances after a crash/restart, closing the
instance-leak on server death. These types are server-internal (``core/``,
not ``api/``): no CLI consumer needs them.

The on-disk body is a JSON envelope — scalar metadata + ``NodeHandle``
tuples as plain JSON, plus ``payload`` (base64 of
``cloudpickle((provider_config, cluster))``) for the two provider-typed
objects, neither of which round-trips as pure JSON (``Cluster.specific``
is arbitrary; ``ProviderConfig`` may hold clients/credentials).
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cloudpickle

from skyward.observability.logger import logger

if TYPE_CHECKING:
    from skyward.core.model import Cluster

SESSIONS_DIR = Path.home() / ".skyward" / "sessions"
SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class NodeHandle:
    node_id: int
    instance_id: str
    ip: str
    private_ip: str | None
    ssh_port: int
    ssh_user: str
    ssh_key_path: str
    ssh_password: str | None
    worker_port: int = 25520


@dataclass(frozen=True, slots=True)
class SessionHandle:
    version: int
    name: str
    created_at: str
    cluster_id: str
    prebaked: bool
    head_node_id: int
    nodes: tuple[NodeHandle, ...]
    payload: bytes  # cloudpickle((provider_config, cluster))


def pack_payload(provider_config: Any, cluster: Cluster[Any]) -> bytes:
    """Serialize ``(provider_config, cluster)`` for a handle's payload."""
    return cloudpickle.dumps((provider_config, cluster))


def unpack_payload(payload: bytes) -> tuple[Any, Any]:
    """Reverse of :func:`pack_payload` → ``(provider_config, cluster)``."""
    return cloudpickle.loads(payload)


def write_handle(handle: SessionHandle, *, sessions_dir: Path = SESSIONS_DIR) -> None:
    """Atomically persist *handle* to ``<sessions_dir>/<name>.json``."""
    sessions_dir.mkdir(parents=True, exist_ok=True)
    envelope = asdict(handle)
    envelope["payload"] = base64.b64encode(handle.payload).decode("ascii")
    target = sessions_dir / f"{handle.name}.json"
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(envelope))
    os.replace(tmp, target)


def read_handle(name: str, *, sessions_dir: Path = SESSIONS_DIR) -> SessionHandle | None:
    """Tolerant read: missing / corrupt / version-mismatch → ``None``."""
    f = sessions_dir / f"{name}.json"
    if not f.exists():
        return None
    try:
        env = json.loads(f.read_text())
        if env.get("version") != SCHEMA_VERSION:
            return None
        return SessionHandle(
            version=env["version"], name=env["name"], created_at=env["created_at"],
            cluster_id=env["cluster_id"], prebaked=env["prebaked"],
            head_node_id=env["head_node_id"],
            nodes=tuple(NodeHandle(**n) for n in env["nodes"]),
            payload=base64.b64decode(env["payload"]),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Ignoring corrupt session handle {f}: {e}", f=str(f), e=repr(exc))
        return None


def list_handles(*, sessions_dir: Path = SESSIONS_DIR) -> tuple[SessionHandle, ...]:
    """Read every ``*.json`` handle in *sessions_dir*, skipping unreadable ones."""
    if not sessions_dir.exists():
        return ()
    handles = (
        read_handle(f.stem, sessions_dir=sessions_dir)
        for f in sorted(sessions_dir.glob("*.json"))
    )
    return tuple(h for h in handles if h is not None)


def remove_handle(name: str, *, sessions_dir: Path = SESSIONS_DIR) -> None:
    """Delete a persisted handle if present."""
    (sessions_dir / f"{name}.json").unlink(missing_ok=True)
