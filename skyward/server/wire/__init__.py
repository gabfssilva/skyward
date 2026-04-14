"""Wire codec: JSON encoding of domain types with discriminated unions.

At import time registers every control-plane discriminated union used by the
HTTP server (:class:`ProviderConfig` subclasses, status/result unions,
session events) plus the named plugin registry.  Consumers import a single
module and the full codec is ready to use.
"""

from __future__ import annotations

from skyward.api.events import Error, Log, Metric, Node, Pool, Scaling, Task
from skyward.api.plugin import Plugin
from skyward.api.provider import ProviderConfig
from skyward.providers import (
    AWS,
    GCP,
    Container,
    Hyperstack,
    JarvisLabs,
    LambdaCloud,
    MassedCompute,
    Novita,
    RunPod,
    Scaleway,
    TensorDock,
    VastAI,
    Verda,
    Vultr,
)
from skyward.server.host.domain import (
    Broadcast,
    CancelledExec,
    Dispatching,
    Failed,
    FailedExec,
    FailedRes,
    GroupMember,
    InterruptedExec,
    InterruptedRes,
    NodeBootstrapping,
    NodeConnecting,
    NodeLost,
    NodeReady,
    NodeWaiting,
    PendingRes,
    Provisioning,
    Queued,
    Ready,
    Run,
    RunningExec,
    RunningRes,
    Stopped,
    Stopping,
    SucceededExec,
    SucceededRes,
)
from skyward.server.wire.codec import (
    from_dict,
    register,
    register_encoder,
    to_dict,
)
from skyward.server.wire.plugins import (
    UnknownPluginTag,
    UnserializablePlugin,
    decode_plugin,
    encode_plugin,
    is_registered_plugin,
    register_plugin,
    registered_plugin_names,
)

__all__ = [
    "UnknownPluginTag",
    "UnserializablePlugin",
    "UnknownWireType",
    "decode_plugin",
    "encode_plugin",
    "from_dict",
    "is_registered_plugin",
    "register",
    "register_encoder",
    "register_plugin",
    "registered_plugin_names",
    "to_dict",
]


# ── Alias for the ValueError raised by the codec on unknown tags ─────
# The B1 codec raises ``ValueError(f"Unknown tag {tag!r} for {target!r}")``.
# We expose a thin subclass for external catch sites to use a single name;
# for now consumers should catch ``ValueError`` — the B1 codec predates this
# registry.  Kept as an alias so future refactors can swap without breaking
# the public wire surface.
UnknownWireType = ValueError


# ── Plugin encode/decode hook ────────────────────────────────────────
register_encoder(Plugin, encode_plugin, decode_plugin)


# ── Providers (ProviderConfig is a Protocol; codec keys on identity) ──
register(
    ProviderConfig,
    aws=AWS,
    gcp=GCP,
    hyperstack=Hyperstack,
    jarvislabs=JarvisLabs,
    **{"lambda": LambdaCloud},
    massed_compute=MassedCompute,
    novita=Novita,
    runpod=RunPod,
    scaleway=Scaleway,
    tensordock=TensorDock,
    vastai=VastAI,
    verda=Verda,
    vultr=Vultr,
    container=Container,
)


# ── ComputeStatus ─────────────────────────────────────────────────────
# Note: Ready/Stopped/Failed collide by name with other module-level
# identifiers elsewhere — the domain module is the authoritative source.
from skyward.server.host.domain import ComputeStatus  # noqa: E402

register(
    ComputeStatus,
    provisioning=Provisioning,
    ready=Ready,
    stopping=Stopping,
    stopped=Stopped,
    failed=Failed,
)


# ── NodeStatus ────────────────────────────────────────────────────────
from skyward.server.host.domain import NodeStatus  # noqa: E402

register(
    NodeStatus,
    waiting=NodeWaiting,
    connecting=NodeConnecting,
    bootstrapping=NodeBootstrapping,
    ready=NodeReady,
    lost=NodeLost,
)


# ── ExecutionStatus ───────────────────────────────────────────────────
from skyward.server.host.domain import ExecutionStatus  # noqa: E402

register(
    ExecutionStatus,
    queued=Queued,
    dispatching=Dispatching,
    running=RunningExec,
    succeeded=SucceededExec,
    failed=FailedExec,
    interrupted=InterruptedExec,
    cancelled=CancelledExec,
)


# ── ResultStatus ──────────────────────────────────────────────────────
from skyward.server.host.domain import ResultStatus  # noqa: E402

register(
    ResultStatus,
    pending=PendingRes,
    running=RunningRes,
    succeeded=SucceededRes,
    failed=FailedRes,
    interrupted=InterruptedRes,
)


# ── TaskExecutionKind ─────────────────────────────────────────────────
from skyward.server.host.domain import TaskExecutionKind  # noqa: E402

register(
    TaskExecutionKind,
    run=Run,
    broadcast=Broadcast,
    group_member=GroupMember,
)


# ── SessionEvent (dotted names, namespace class -> event kwargs) ──────
from skyward.api.events import SessionEvent  # noqa: E402

register(
    SessionEvent,
    **{
        "Pool.Provisioning": Pool.Provisioning,
        "Pool.PhaseChanged": Pool.PhaseChanged,
        "Pool.Stopped": Pool.Stopped,
        "Pool.Reconciled": Pool.Reconciled,
        "Pool.Provisioned": Pool.Provisioned,
        "Pool.ProvisionFailed": Pool.ProvisionFailed,
        "Node.Connected": Node.Connected,
        "Node.Ready": Node.Ready,
        "Node.Lost": Node.Lost,
        "Node.ConnectionFailed": Node.ConnectionFailed,
        "Node.Preempted": Node.Preempted,
        "Node.WorkerFailed": Node.WorkerFailed,
        "Node.Bootstrap.Started": Node.Bootstrap.Started,
        "Node.Bootstrap.Completed": Node.Bootstrap.Completed,
        "Node.Bootstrap.Output": Node.Bootstrap.Output,
        "Node.Bootstrap.Done": Node.Bootstrap.Done,
        "Node.Bootstrap.Failed": Node.Bootstrap.Failed,
        "Node.Bootstrap.Command": Node.Bootstrap.Command,
        "Task.Queued": Task.Queued,
        "Task.Assigned": Task.Assigned,
        "Task.Completed": Task.Completed,
        "Task.Failed": Task.Failed,
        "Task.BroadcastPartial": Task.BroadcastPartial,
        "Metric.Sampled": Metric.Sampled,
        "Log.Emitted": Log.Emitted,
        "Scaling.DesiredChanged": Scaling.DesiredChanged,
        "Scaling.Spawning": Scaling.Spawning,
        "Scaling.Draining": Scaling.Draining,
        "Scaling.DrainCompleted": Scaling.DrainCompleted,
        "Error.Occurred": Error.Occurred,
    },
)


# ── Built-in plugins ─────────────────────────────────────────────────
from skyward.plugins.cuml import cuml as _cuml  # noqa: E402
from skyward.plugins.jax import jax as _jax  # noqa: E402
from skyward.plugins.joblib import joblib as _joblib  # noqa: E402
from skyward.plugins.keras import keras as _keras  # noqa: E402
from skyward.plugins.mig import mig as _mig  # noqa: E402
from skyward.plugins.mps import mps as _mps  # noqa: E402
from skyward.plugins.sklearn import sklearn as _sklearn  # noqa: E402
from skyward.plugins.torch import torch as _torch  # noqa: E402

register_plugin("torch", _torch)
register_plugin("jax", _jax)
register_plugin("keras", _keras)
register_plugin("cuml", _cuml)
register_plugin("joblib", _joblib)
register_plugin("sklearn", _sklearn)
register_plugin("mig", _mig)
register_plugin("mps", _mps)


def _hydrate_forward_refs() -> None:
    """Materialize ``TYPE_CHECKING`` names on ``skyward.api.spec``.

    ``skyward.api.spec`` uses ``from __future__ import annotations`` so every
    dataclass field annotation is a string.  The wire codec calls
    ``typing.get_type_hints(cls)`` during decode and fails with ``NameError``
    because ``ProviderConfig``, ``Plugin``, ``Accelerator``, etc. are only
    imported under ``TYPE_CHECKING``.  Attaching the real classes to the
    module globals lets ``get_type_hints`` resolve so the codec walks into
    nested dataclasses instead of returning raw dicts.
    """
    import skyward.api.spec as spec_mod

    if getattr(spec_mod, "_skyward_forward_refs_hydrated", False):
        return
    from skyward.accelerators import Accelerator
    from skyward.actors.messages import ProviderName
    from skyward.api.logging import LogConfig
    from skyward.storage import Storage

    spec_mod.Accelerator = Accelerator  # type: ignore[attr-defined]
    spec_mod.ProviderName = ProviderName  # type: ignore[attr-defined]
    spec_mod.LogConfig = LogConfig  # type: ignore[attr-defined]
    spec_mod.Plugin = Plugin  # type: ignore[attr-defined]
    spec_mod.ProviderConfig = ProviderConfig  # type: ignore[attr-defined]
    spec_mod.Storage = Storage  # type: ignore[attr-defined]
    spec_mod._skyward_forward_refs_hydrated = True  # type: ignore[attr-defined]


_hydrate_forward_refs()
