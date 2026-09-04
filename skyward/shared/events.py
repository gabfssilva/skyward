"""What the daemon says happened, one struct per fact.

An event is a fact: something happened, once, at a moment. A compute becoming
ready is a fact; so is an offer being chosen, a region refusing to sell, a machine
being bought at a price. The first kind moves a state and the second kind does not,
and both are said here in the same voice, because a watcher wants to know what
happened rather than which table it landed in. Which facts move which states is
:mod:`skyward.shared.lifecycle`'s to say, not the event's.

Every struct carries ``compute``; the ones about a node carry ``node``, the ones
about a task carry ``task``. The ``type`` tag inside the payload is what makes it
decodable on its own — written to a file, replayed by ``sky log export``, or handed
to a client that never saw the SSE frame that carried it.
"""

from __future__ import annotations

from datetime import datetime

from msgspec import Struct

from skyward.shared.schemas import ErrorCode, Market, NodeState, PhaseMark, TaskEventState


class ComputeCreated(Struct, frozen=True, tag_field="type", tag="compute.created"):
    """The definition was accepted and written down. The compute exists, and owns nothing yet."""

    compute: str


class ComputeBound(Struct, frozen=True, tag_field="type", tag="compute.bound"):
    """The compute was given an address in the world: an offer, a region, the markets to buy on.

    ``previous`` is the offer it was bound to before, when a region refused to sell
    and the compute followed the next offer that fits. Said once per binding, which
    is the moment the price of every machine it will buy is decided.
    """

    compute: str
    offer: str
    instance_type: str
    region: str | None
    markets: tuple[Market, ...]
    previous: str | None = None


class ComputeAdopted(Struct, frozen=True, tag_field="type", tag="compute.adopted"):
    """Another daemon bound this compute first, and this one is carrying on under its binding."""

    compute: str


class ComputeProvisioning(Struct, frozen=True, tag_field="type", tag="compute.provisioning"):
    """Fewer machines answer than the floor asks for, and the reconciler is closing the gap.

    Said on the way up and again on the way back down: a pool that lost a machine
    or grew its floor is provisioning again, and a watcher that only heard about
    the first time would think it was still ready.
    """

    compute: str
    nodes_ready: int
    nodes_total: int
    generation: int


class ComputeReady(Struct, frozen=True, tag_field="type", tag="compute.ready"):
    """Enough machines answer to satisfy the floor. Work can be placed."""

    compute: str
    nodes_ready: int
    nodes_total: int
    generation: int


class ComputeDegraded(Struct, frozen=True, tag_field="type", tag="compute.degraded"):
    """A reconcile pass broke on this compute, and the next pass will try again.

    Not a terminal failure — there is none for a compute. Said once when the state
    moves, not once per failing tick. ``code`` is the failure's own when it had one,
    and ``reconcile_failed`` when all the pass caught was an exception.
    """

    compute: str
    error: str
    code: ErrorCode = "reconcile_failed"


class GenerationCreated(Struct, frozen=True, tag_field="type", tag="compute.generation.created"):
    """A new definition was frozen: a resize, or an earlier generation brought back."""

    compute: str
    number: int


class GenerationApplied(Struct, frozen=True, tag_field="type", tag="compute.generation.applied"):
    """The machines now reflect this definition."""

    compute: str
    number: int


class LeaseClaimed(Struct, frozen=True, tag_field="type", tag="compute.lease.claimed"):
    """A process took ownership. A renewal by the same owner is not said again."""

    compute: str
    owner: str


class LeaseReleased(Struct, frozen=True, tag_field="type", tag="compute.lease.released"):
    """The owner let go without asking for anything to be destroyed."""

    compute: str


class ComputeAbandoned(Struct, frozen=True, tag_field="type", tag="compute.abandoned"):
    """Nothing renewed the lease and ``delete_on_exit`` was set, so it is going away.

    Its own fact rather than a compute state, because a lapsed lease is not a
    failure — it is what a client exiting looks like from in here.
    """

    compute: str


class ComputeDeleting(Struct, frozen=True, tag_field="type", tag="compute.deleting"):
    """Destruction was asked for, and the machines are being given back.

    Said when the intent is written, by whoever wrote it. The reconciler says it
    again on every pass with the machines that remain, and the repetitions move
    nothing: they carry the count down while the state stays where it is.
    """

    compute: str
    nodes_ready: int
    nodes_total: int


class ComputeDeletionFailed(Struct, frozen=True, tag_field="type", tag="compute.deletion_failed"):
    """A pass during teardown broke, and the next one will carry on giving the machines back.

    A fact, not a state: a compute being deleted has nowhere to be degraded *to*.
    Said once per distinct failure — the same error on the next tick is a repeat
    and is not written again. ``release_pending`` is the provider refusing to
    release the binding while every machine is already gone.
    """

    compute: str
    error: str
    code: ErrorCode = "reconcile_failed"


class StraysTerminated(Struct, frozen=True, tag_field="type", tag="compute.strays_terminated"):
    """Machines the provider held under this compute that no row of it owns were terminated."""

    compute: str
    machines: tuple[str, ...]


class ComputeDeleted(Struct, frozen=True, tag_field="type", tag="compute.deleted"):
    """Every machine is gone and the binding is released. Nothing bills any more.

    The counts are what it observed — none — spelled out so that the row's status
    is a projection of the event like every other transition's.
    """

    compute: str
    nodes_ready: int = 0
    nodes_total: int = 0


class CostEvent(Struct, frozen=True, tag_field="type", tag="compute.cost"):
    """What the compute has accrued so far, and over how many live machines.

    Published rather than recorded: a gauge sampled every few seconds has no
    replay value, and the event log has no GC to save it from one.
    """

    compute: str
    cost: float
    nodes: int
    at: datetime


class NodeEvent(Struct, frozen=True, tag_field="type", tag="node.state"):
    """One machine's lifecycle moved, carried by every ``node.{state}``.

    ``state`` repeats the event name because a payload that has been written
    down, exported, or replayed out of the stream has to say what it is without
    the frame that carried it.
    """

    compute: str
    node: str
    state: NodeState
    error: str | None = None


class ProgressEvent(Struct, frozen=True, tag_field="type", tag="node.progress"):
    """What a machine short of an address is doing, while it is still doing it.

    Published rather than recorded, like a gauge: a percentage is true for the
    moment it is sent, and a late subscriber wants where the machine got to, not
    the hundred readings it passed through while nobody was watching. What outlives
    the wait is the node's state and, if it never arrives, the reason it was given
    up on.

    ``progress`` is what the machine is doing and ``completion`` is how far into it,
    kept apart so that a reader can draw the fraction instead of spelling it: a
    terminal gets a bar that fills, a log file gets the same words it always got.
    """

    compute: str
    node: str
    progress: str
    completion: float | None = None


def progressed(progress: str, completion: float | None) -> str:
    """What a machine is doing and how far into it, as one phrase.

    One rendering, because the same words are the line a log carries and the reason
    a machine that never moved again was given up on, and a percentage written two
    ways reads as two different machines.
    """
    return progress if completion is None else f"{progress} ({completion * 100:.0f}%)"


class ConsoleEvent(Struct, frozen=True, tag_field="type", tag="node.console"):
    """A line a node printed, and the work it belongs to when it belongs to some.

    Recorded, because output that only existed live would be output a client that
    reconnected could never see.

    ``task`` carries the *execution* — the attempt on this node — because that is
    what the machine that wrote the line was handed. A reader after a whole task's
    output wants every execution of it, not a string equal to the task's id.
    """

    compute: str
    node: str
    content: str
    task: str | None = None


class PhaseEvent(Struct, frozen=True, tag_field="type", tag="node.phase"):
    """A bootstrap phase turning over, so a late subscriber replays the checklist.

    ``phase`` names the step; ``event`` says whether it opened, closed, or broke.
    """

    compute: str
    node: str
    event: PhaseMark
    phase: str
    at: datetime
    error: str | None = None


class MetricEvent(Struct, frozen=True, tag_field="type", tag="node.metrics"):
    """One gauge reading off one node. Published rather than recorded, like ``compute.cost``."""

    compute: str
    node: str
    name: str
    value: float


class TaskEvent(Struct, frozen=True, tag_field="type", tag="task.state"):
    """A task began, or reached the one terminal outcome it is allowed."""

    compute: str
    task: str
    state: TaskEventState


type Event = (
    ComputeCreated
    | ComputeBound
    | ComputeAdopted
    | ComputeProvisioning
    | ComputeReady
    | ComputeDegraded
    | GenerationCreated
    | GenerationApplied
    | LeaseClaimed
    | LeaseReleased
    | ComputeAbandoned
    | ComputeDeleting
    | ComputeDeletionFailed
    | StraysTerminated
    | ComputeDeleted
    | CostEvent
    | NodeEvent
    | ProgressEvent
    | ConsoleEvent
    | PhaseEvent
    | MetricEvent
    | TaskEvent
)
"""Everything the SSE stream carries, as one tagged union.

The SSE frame's ``event:`` field is the name a subscriber filters on. For a
compute it is the ``type`` tag itself, one name per fact; for a node or a task
it is still finer than the tag — ten node states and four task outcomes share a
struct each — which is what :func:`name` is for. Flat on purpose: the OpenAPI
discriminator is read off this union's members, and a union nested in it would
be one member it cannot name.
"""

def name(event: Event) -> str:
    """The frame name an event goes out under, and the name a subscriber filters on."""
    match event:
        case NodeEvent(state=state):
            return f"node.{state}"
        case TaskEvent(state=state):
            return f"task.{state}"
        case _:
            return str(type(event).__struct_config__.tag)


__all__ = [
    "ComputeAbandoned",
    "ComputeAdopted",
    "ComputeBound",
    "ComputeCreated",
    "ComputeDegraded",
    "ComputeDeleted",
    "ComputeDeleting",
    "ComputeDeletionFailed",
    "ComputeProvisioning",
    "ComputeReady",
    "ConsoleEvent",
    "CostEvent",
    "Event",
    "GenerationApplied",
    "GenerationCreated",
    "LeaseClaimed",
    "LeaseReleased",
    "MetricEvent",
    "NodeEvent",
    "PhaseEvent",
    "ProgressEvent",
    "StraysTerminated",
    "TaskEvent",
    "name",
    "progressed",
]
