from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Final

from casty import ActorRef

from skyward.actors.messages import HeadAddressKnown, NodeInstance

type NodeId = int

_PYTHON_VERSION: Final = f"{sys.version_info.major}.{sys.version_info.minor}"


class PythonVersionMismatchError(RuntimeError):
    def __init__(self, local: str, remote: str) -> None:
        self.local = local
        self.remote = remote
        super().__init__(
            f"Python version mismatch: local={local}, remote={remote}. "
            f"Cloudpickle cannot safely serialize bytecode across versions. "
            f"Set Image(python='{local}') or use Image(python='auto')."
        )


def check_python_version(remote_version: str) -> None:
    if remote_version != _PYTHON_VERSION:
        raise PythonVersionMismatchError(local=_PYTHON_VERSION, remote=remote_version)


@dataclass(frozen=True, slots=True)
class PendingTask:
    fn: Any
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    reply_to: ActorRef[Any]
    task_id: str
    timeout: float


@dataclass(frozen=True, slots=True)
class NodeState:
    cluster: Any
    provider: Any
    ni: NodeInstance | None = None
    transport_ref: ActorRef | None = None
    local_port: int = 0
    head_info: HeadAddressKnown | None = None
    pending_tasks: tuple[PendingTask, ...] = ()
    client: Any = None
    worker_ref: ActorRef | None = None
    pool_info_json: str = ""
    env_vars: dict[str, str] = field(default_factory=dict)
    around_app_hooks: tuple[tuple[str, Any], ...] = ()
    around_process_hooks: tuple[tuple[str, Any], ...] = ()
    inflight: dict[str, ActorRef] = field(default_factory=dict)
    task_counter: int = 0
