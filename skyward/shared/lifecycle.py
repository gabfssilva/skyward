"""Which facts move which states: the state machines, as tables.

One table per entity, one arrow per row: the event, where it may come from, where
it leads. The reconciler decides *that* a compute is ready by counting machines;
this is what says what that decision does to the row, and it is the same answer
whether the daemon is applying the event to its store or a client is folding it
into the view it hands a callback. Two readers, one table, so the two cannot
disagree about what an event means.

An event that is in no table is a fact that moves nothing — an offer chosen, a
cost sampled, a line printed — and applying it writes it down and changes no
state. An event that leads to the state the entity is already in is a repeat: the
reconciler runs the same pass many times by design, and saying ``ready`` to a ready
compute is silence, not an error. An event whose origin the table does not admit is
illegal, and refused: ``deleted`` has no way back.
"""

from __future__ import annotations

from collections.abc import Mapping

from skyward.shared.errors import IllegalTransitionError
from skyward.shared.events import (
    ComputeDegraded,
    ComputeDeleted,
    ComputeDeleting,
    ComputeProvisioning,
    ComputeReady,
    Event,
)
from skyward.shared.schemas import ComputeState

type Arrow[S] = tuple[frozenset[S], S]
"""Where an event may come from, and where it leads."""

OPEN: frozenset[ComputeState] = frozenset({"requested", "provisioning", "ready", "degraded"})
"""The states a compute reconciles forward from. Everything before deletion was asked for."""

CLOSING: frozenset[ComputeState] = frozenset({"deleting"})
"""The one state between asking for destruction and the provider confirming it."""

COMPUTE: Mapping[type[Event], Arrow[ComputeState]] = {
    ComputeProvisioning: (OPEN, "provisioning"),
    ComputeReady: (OPEN, "ready"),
    ComputeDegraded: (OPEN, "degraded"),
    ComputeDeleting: (OPEN | CLOSING, "deleting"),
    ComputeDeleted: (CLOSING, "deleted"),
}
"""The compute's arrows. ``requested`` is where a row is born, and no event leads back to it."""


def compute(state: ComputeState, event: Event) -> ComputeState | None:
    """The state a compute is in after ``event``, or ``None`` when the event moves nothing.

    Raises :class:`IllegalTransitionError` when the table has no arrow from ``state``
    for this event. Leading to the state it is already in is not a transition and
    not an error: the caller sees the same state back and knows nothing moved.
    """
    return _apply(COMPUTE, state, event)


def leads(event: Event) -> ComputeState | None:
    """Where a compute event leads, whatever it came from.

    The client's reading of the table. A watcher folds a replay that starts at the
    log's beginning into a view it may already have hydrated from the API, so the
    origin it holds is not the origin the daemon checked against — it takes the
    destination on trust, because the daemon already refused what was illegal.
    """
    return arrow[1] if (arrow := COMPUTE.get(type(event))) else None


def _apply[S: str](table: Mapping[type[Event], Arrow[S]], state: S, event: Event) -> S | None:
    if (arrow := table.get(type(event))) is None:
        return None
    sources, target = arrow
    if state == target:
        return state
    if state not in sources:
        raise IllegalTransitionError(f"{type(event).__name__} has no arrow from {state}", state=state, event=type(event).__name__)
    return target


__all__ = ["CLOSING", "COMPUTE", "OPEN", "Arrow", "compute", "leads"]
