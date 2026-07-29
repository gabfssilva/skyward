"""What kind of machine, and how many.

A spec is one way to satisfy a pool: this provider, this accelerator, this
shape. A pool may be given several — "an A100 on AWS, or an A100 on Vast, take
whichever is cheaper" — and the control plane picks among the offers they match.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from skyward.accelerators import Accelerator
from skyward.protocol.architectures import Architecture
from skyward.protocol.schemas import NodeBounds as Nodes
from skyward.sdk.provider import Provider

if TYPE_CHECKING:
    from skyward.runtime.api import Info
    from skyward.storage import Storage

type NodeSpec = int | tuple[int, int] | Nodes
type ExecutorType = Literal["thread", "process", "loky"]
type Route = Literal["round_robin"]


@dataclass(frozen=True, slots=True)
class HealthChecker:
    fn: Callable[[Info], bool | str]
    interval: float = 30.0
    timeout: float = 15.0
    consecutive_failures: int = 3
    initial_delay: float = 0.0

    def __post_init__(self) -> None:
        if self.interval <= 0:
            raise ValueError(f"interval must be > 0, got {self.interval}")
        if self.timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {self.timeout}")
        if self.consecutive_failures < 1:
            raise ValueError(f"consecutive_failures must be >= 1, got {self.consecutive_failures}")
        if self.initial_delay < 0:
            raise ValueError(f"initial_delay must be >= 0, got {self.initial_delay}")


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
    disk_gb: int | None = None
    architecture: Architecture | None = None
    max_hourly_cost: float | None = None


@dataclass(frozen=True, slots=True)
class Port:
    """Expose a node port on a fixed local port.

    Each connection to ``127.0.0.1:<local>`` is bridged to ``remote`` on a ready
    node, chosen by ``route``, over that node's existing SSH connection. The local
    listener binds loopback only.

    Attributes
    ----------
    remote : int
        The port the service listens on inside the node.
    local : int
        The local port to bind. ``0`` lets the OS choose one.
    route : Route
        How connections are spread across the ready nodes.

    Examples
    --------
    >>> with sky.Compute(provider=sky.AWS(), nodes=2, ports=[sky.Port(remote=8080, local=8080)]) as pool:
    ...     serve() @ pool  # a request to 127.0.0.1:8080 round-robins across the nodes
    """

    remote: int
    local: int = 0
    route: Route = "round_robin"


@dataclass(frozen=True, slots=True)
class Options:
    """Operational tuning for a compute — timeouts, retries, autoscaling.

    Sensible defaults reproduce the runtime's built-in behavior, so most pools never
    construct one. The daemon-side knobs are carried to the control plane on the
    spec; the two session timeouts (``ready_timeout``, ``shutdown_timeout``) stay in
    this process, because they govern how long *this* client waits for its own pool
    and never leave it.

    Parameters
    ----------
    ssh_timeout : float
        Seconds to keep dialing a machine before giving up on reaching it.
    provision_retry_delay : float
        Seconds between connection attempts to a machine still coming up.
    max_provision_attempts : int
        How many times a dropped connection is redialed before the node is lost.
    worker_timeout : float
        Seconds to wait for the worker to come up once bootstrap has finished.
    autoscale_idle_timeout : float
        Seconds a node must sit idle before an elastic pool may reclaim it.
    autoscale_cooldown : float
        Seconds between autoscaling decisions. ``0`` is no cooldown.
    default_compute_timeout : float
        Seconds a task may run when it names no deadline of its own. ``0`` is unbounded.
    health_command : str | None
        A shell command run on each node to ask whether the machine is still usable.
        ``None`` probes nothing.
    health_interval : float
        Seconds between health probes.
    health_failures : int
        How many consecutive failed probes make a node lost, and so replaced.
    ready_timeout : float
        Seconds to wait for the pool to become ready before giving up.
    shutdown_timeout : float
        Seconds to wait for the pool to finish deleting on exit.

    Examples
    --------
    >>> with sky.Compute(provider=sky.AWS(), options=sky.Options(ssh_timeout=120)) as pool:
    ...     train(data) >> pool
    """

    ssh_timeout: float = 300.0
    provision_retry_delay: float = 2.0
    max_provision_attempts: int = 30
    worker_timeout: float = 180.0
    autoscale_idle_timeout: float = 30.0
    autoscale_cooldown: float = 0.0
    default_compute_timeout: float = 0.0
    health_command: str | None = None
    health_interval: float = 30.0
    health_failures: int = 3
    health_checker: HealthChecker | None = None
    cluster: bool | None = None
    ready_timeout: float = 900.0
    shutdown_timeout: float = 300.0


SYSTEM_PATHS = frozenset({"/", "/opt", "/opt/skyward", "/root", "/tmp"})
"""Where a mount would take the machine down with it.

``/opt/skyward`` is the venv, the bootstrap script and the journal the daemon
reads the node's life from; ``/`` and ``/root`` are the machine. Shadowing any of
them with a bucket is not a mount that fails, it is a node that never reports
again.
"""


@dataclass(frozen=True, slots=True)
class Volume:
    """A bucket every node mounts, at a path you choose.

    The nodes see a directory: ordinary ``open`` and ``os.listdir`` work, and a
    dataset too big to ship as user code is read straight off object storage
    instead. It is a network filesystem wearing a directory's clothes, so it is
    read-heavy by design — ``read_only`` is the default for that reason.

    Where the credentials come from is what ``storage`` decides. Left ``None``,
    the daemon resolves them from the provider the compute was bought on, and for
    a bucket in that same account nothing secret is created or transmitted at all.
    Set, they are yours — an R2 or Backblaze bucket the daemon has no account for —
    and they travel to the daemon as a blob rather than on the spec, so they are
    never served back by the compute API.

    A compute takes one kind or the other, never a mix: two sets of credentials
    under one set of mounts is a compute nobody can say who is paying for.

    Attributes
    ----------
    bucket : str
        The bucket to mount. On RunPod, which attaches storage instead of mounting
        it, the id or name of a network volume.
    mount : str
        The absolute path the bucket appears at on every node.
    prefix : str
        A subdirectory of the bucket to mount, rather than its root.
    read_only : bool
        Whether writes are refused. Buckets shared by several volumes are mounted
        writable if any of them asked to write.
    storage : Storage | None
        Credentials for buckets the provider cannot reach on its own.

    Examples
    --------
    >>> with sky.Compute(provider=sky.AWS(), volumes=[sky.Volume(bucket="training-data", mount="/data")]) as pool:
    ...     train() >> pool  # the node reads /data/*.parquet
    """

    bucket: str
    mount: str
    prefix: str = ""
    read_only: bool = True
    storage: Storage | None = None

    def __post_init__(self) -> None:
        if not self.mount.startswith("/"):
            raise ValueError(f"a volume mounts at an absolute path, not {self.mount!r}")
        if self.mount.rstrip("/") in SYSTEM_PATHS or self.mount == "/":
            raise ValueError(f"{self.mount!r} is the machine's own, and mounting over it would take the node down")


__all__ = ["Accelerator", "Architecture", "Executor", "ExecutorType", "NodeSpec", "Nodes", "Options", "Port", "Route", "Spec", "Volume"]
