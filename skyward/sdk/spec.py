"""What kind of machine, and how many.

A spec is one way to satisfy a pool: this provider, this accelerator, this
shape. A pool may be given several — "an A100 on AWS, or an A100 on Vast, take
whichever is cheaper" — and the control plane picks among the offers they match.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from skyward.accelerators import Accelerator
from skyward.protocol.schemas import NodeBounds as Nodes
from skyward.sdk.provider import Provider

type NodeSpec = int | tuple[int, int] | Nodes
type ExecutorType = Literal["thread", "process", "loky"]


@dataclass(frozen=True, slots=True)
class Executor:
    """How the tasks run on the machine: where, how many, and how far ahead.

    ``thread`` runs tasks on a bounded thread pool — the default, and the only
    one that shares the worker's own address space, so the distributed collections
    reach the cluster with nothing in between. ``process`` and ``loky`` run each
    task in a subprocess, which is what a task that holds the GIL or leaks state
    wants; they reach the collections over a bridge back to the worker.

    ``reuse`` is a process knob and nothing else: a ``process`` pool with
    ``reuse=False`` spends one subprocess per task and throws it away, which is the
    clean-slate every time. ``reuse=True`` keeps the subprocesses between tasks, and
    ``loky`` is the reusable pool that also restarts a worker that died — so ``reuse``
    does not apply to it, nor to ``thread``, whose threads are always reused.

    ``concurrency`` is the pool's width — how many tasks run at once. ``buffer`` is
    the slack above it: that many more tasks are admitted and their payloads made
    ready, so a slot that frees finds the next one in hand rather than a round trip
    away. It is also the depth the daemon reads as backpressure before it grows the
    compute.

    Attributes
    ----------
    type : {"thread", "process", "loky"}
        The backend the tasks run on.
    reuse : bool
        Whether subprocesses live between tasks. Only meaningful for ``process``.
    concurrency : int | None
        How many tasks run at once. ``None`` is one.
    buffer : int
        How many more tasks to admit and keep ready above ``concurrency``.
    """

    type: ExecutorType = "thread"
    reuse: bool = True
    concurrency: int | None = None
    buffer: int = 0

    def __post_init__(self) -> None:
        if not self.reuse and self.type != "process":
            raise ValueError(f"reuse=False only applies to executor='process', not {self.type!r}")


@dataclass(frozen=True, slots=True)
class Spec:
    provider: Provider
    accelerator: str | Accelerator | None = None
    cpus: int | None = None
    memory_gb: int | None = None
    region: str | None = None


__all__ = ["Accelerator", "Executor", "ExecutorType", "NodeSpec", "Nodes", "Spec"]
