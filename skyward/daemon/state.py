"""Event-sourced daemon state -- persists pool lifecycle via Casty journal.

On daemon restart, the journal replays events to reconstruct which pools
were alive, enabling crash recovery via provider.get_instance().
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from casty import ActorRef, Behavior, Behaviors, EventJournal, SnapshotEvery

# -- State -----------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PoolEntry:
    """Persisted metadata for one daemon-managed pool."""
    cluster_id: str
    instance_ids: tuple[str, ...]
    clients: frozenset[str] = frozenset()
    provider_name: str = ""
    cluster_bytes: bytes = b""
    spec_bytes: bytes = b""
    provider_config_bytes: bytes = b""


@dataclass(frozen=True, slots=True)
class DaemonState:
    """Aggregate state derived from events."""
    pools: MappingProxyType[str, PoolEntry] = field(
        default_factory=lambda: MappingProxyType({}),
    )


# -- Events (persisted facts) ---------------------------------------------

@dataclass(frozen=True, slots=True)
class PoolRegistered:
    pool_name: str
    cluster_id: str
    instance_ids: tuple[str, ...]
    provider_name: str = ""
    cluster_bytes: bytes = b""
    spec_bytes: bytes = b""
    provider_config_bytes: bytes = b""


@dataclass(frozen=True, slots=True)
class PoolRemoved:
    pool_name: str


@dataclass(frozen=True, slots=True)
class ClientJoined:
    pool_name: str
    client_id: str


@dataclass(frozen=True, slots=True)
class ClientLeft:
    pool_name: str
    client_id: str


type DaemonEvent = PoolRegistered | PoolRemoved | ClientJoined | ClientLeft


# -- Commands (requests) ---------------------------------------------------

@dataclass(frozen=True, slots=True)
class RegisterPool:
    pool_name: str
    cluster_id: str
    instance_ids: tuple[str, ...]
    reply_to: ActorRef[Any]
    provider_name: str = ""
    cluster_bytes: bytes = b""
    spec_bytes: bytes = b""
    provider_config_bytes: bytes = b""


@dataclass(frozen=True, slots=True)
class RemovePool:
    pool_name: str
    reply_to: ActorRef[Any]


@dataclass(frozen=True, slots=True)
class AddClient:
    pool_name: str
    client_id: str
    reply_to: ActorRef[Any]


@dataclass(frozen=True, slots=True)
class RemoveClient:
    pool_name: str
    client_id: str
    reply_to: ActorRef[Any]


@dataclass(frozen=True, slots=True)
class GetState:
    reply_to: ActorRef[DaemonState]


type DaemonCommand = RegisterPool | RemovePool | AddClient | RemoveClient | GetState


# -- Event handler (pure) -------------------------------------------------

def apply_event(state: DaemonState, event: DaemonEvent) -> DaemonState:
    """Apply one event to derive new state. Pure, synchronous."""
    match event:
        case PoolRegistered(
            pool_name=name, cluster_id=cid, instance_ids=iids,
            provider_name=pn, cluster_bytes=cb,
            spec_bytes=sb, provider_config_bytes=pcb,
        ):
            entry = PoolEntry(
                cluster_id=cid, instance_ids=iids,
                provider_name=pn, cluster_bytes=cb, spec_bytes=sb,
                provider_config_bytes=pcb,
            )
            return DaemonState(pools=MappingProxyType({**state.pools, name: entry}))

        case PoolRemoved(pool_name=name):
            remaining = {k: v for k, v in state.pools.items() if k != name}
            return DaemonState(pools=MappingProxyType(remaining))

        case ClientJoined(pool_name=name, client_id=cid):
            if name not in state.pools:
                return state
            entry = state.pools[name]
            updated = replace(entry, clients=entry.clients | {cid})
            return DaemonState(pools=MappingProxyType({**state.pools, name: updated}))

        case ClientLeft(pool_name=name, client_id=cid):
            if name not in state.pools:
                return state
            entry = state.pools[name]
            updated = replace(entry, clients=entry.clients - {cid})
            return DaemonState(pools=MappingProxyType({**state.pools, name: updated}))

    return state


# -- Command handler (async) -----------------------------------------------

async def _on_command(
    ctx: Any, state: DaemonState, cmd: DaemonCommand,
) -> Any:
    """Decide which events to persist based on commands."""
    match cmd:
        case RegisterPool(
            pool_name=name, cluster_id=cid, instance_ids=iids,
            reply_to=r, provider_name=pn,
            cluster_bytes=cb, spec_bytes=sb, provider_config_bytes=pcb,
        ):
            r.tell(True)
            return Behaviors.persisted(events=[
                PoolRegistered(
                    pool_name=name, cluster_id=cid, instance_ids=iids,
                    provider_name=pn, cluster_bytes=cb,
                    spec_bytes=sb, provider_config_bytes=pcb,
                ),
            ])

        case RemovePool(pool_name=name, reply_to=r):
            r.tell(True)
            return Behaviors.persisted(events=[PoolRemoved(pool_name=name)])

        case AddClient(pool_name=name, client_id=cid, reply_to=r):
            r.tell(True)
            return Behaviors.persisted(events=[ClientJoined(pool_name=name, client_id=cid)])

        case RemoveClient(pool_name=name, client_id=cid, reply_to=r):
            r.tell(True)
            return Behaviors.persisted(events=[ClientLeft(pool_name=name, client_id=cid)])

        case GetState(reply_to=r):
            r.tell(state)
            return Behaviors.same()

    return Behaviors.unhandled()


# -- Actor factory ---------------------------------------------------------

def daemon_state_actor(
    entity_id: str, journal: EventJournal,
) -> Behavior[DaemonCommand]:
    """Create an event-sourced actor that tracks daemon pool lifecycle."""
    return Behaviors.event_sourced(
        entity_id=entity_id,
        journal=journal,
        initial_state=DaemonState(),
        on_event=apply_event,
        on_command=_on_command,
        snapshot_policy=SnapshotEvery(n_events=50),
    )
