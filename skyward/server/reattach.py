"""Server-side reattach: persist pool handles and re-adopt them on boot.

Persisting closes the instance-leak on server death; re-adopting on boot
restores ``sky stop/status/run/log`` over the still-live instances. The
handle is written when a pool reaches ready and re-written on scale
events. On boot, each persisted handle is re-adopted; a handle whose
instances are gone is cleaned up (cluster torn down + handle removed).
"""

from __future__ import annotations

from contextlib import suppress
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from skyward.core.session_store import (
    SCHEMA_VERSION,
    NodeHandle,
    SessionHandle,
    list_handles,
    pack_payload,
    remove_handle,
    unpack_payload,
    write_handle,
)
from skyward.observability.logger import logger

if TYPE_CHECKING:
    from skyward.server.state import ServerState


def _node_handle(ns: Any, inst: Any, cluster: Any) -> NodeHandle:
    return NodeHandle(
        node_id=ns.node_id,
        instance_id=ns.instance_id,
        ip=inst.ip or "",
        private_ip=inst.private_ip,
        ssh_port=inst.ssh_port,
        ssh_user=cluster.ssh_user,
        ssh_key_path=cluster.ssh_key_path,
        ssh_password=inst.ssh_password,
    )


def persist_handle(name: str, provider_config: Any, pool: Any) -> None:
    """Write the reattach handle for a ready pool (best-effort).

    Reads the pool's live snapshot for cluster + per-node SSH coordinates;
    ``provider_config`` is the ``ProviderConfig`` needed to recreate the
    provider on reattach. Safe to call from a worker thread.
    """
    from skyward.core.pool import ComputePool

    if not isinstance(pool, ComputePool):
        return
    snap = pool.snapshot()
    cluster = snap.cluster
    if cluster is None:
        return
    by_id = {i.id: i for i in snap.instances}
    nodes = tuple(
        _node_handle(ns, by_id[ns.instance_id], cluster)
        for ns in sorted(snap.nodes, key=lambda n: n.node_id)
        if ns.instance_id in by_id
    )
    if not nodes:
        return
    write_handle(SessionHandle(
        version=SCHEMA_VERSION,
        name=name,
        created_at=datetime.now(UTC).isoformat(),
        cluster_id=cluster.id,
        prebaked=cluster.prebaked,
        head_node_id=0,
        nodes=nodes,
        payload=pack_payload(provider_config, cluster),
    ))


def subscribe_repersist(
    state: ServerState, name: str, provider_config: Any, pool: Any,
) -> None:
    """Re-write the handle on ``Node.Ready``/``Node.Lost`` for this pool.

    Persistence runs on the broadcast executor — never on the event-loop
    thread, where ``pool.snapshot()`` (a blocking ask) would deadlock.
    """
    from skyward.api.events import Node

    def _on_scale(event: Any) -> None:
        if getattr(event, "pool_name", None) != name:
            return
        if isinstance(event, (Node.Ready, Node.Lost)):
            state.broadcast_executor.submit(persist_handle, name, provider_config, pool)

    state.reattach_unsubs[name] = state.session.projection.subscribe(on_event=_on_scale)


def drop_persistence(state: ServerState, name: str) -> None:
    """Unsubscribe the re-persist tap and delete the on-disk handle."""
    unsub = state.reattach_unsubs.pop(name, None)
    if unsub is not None:
        with suppress(Exception):
            unsub()
    remove_handle(name)


def reattach_pools(state: ServerState) -> None:
    """Re-adopt every persisted pool at server boot; clean up dead ones."""
    session = state.session
    for handle in list_handles():
        try:
            provider_config, cluster = unpack_payload(handle.payload)
            # Drive from handle.nodes (one per rank), matching each to a unique
            # cluster instance by id — cluster.instances may carry duplicates.
            by_id = {inst.id: inst for inst in cluster.instances}
            adopted = [(n.node_id, by_id[n.instance_id]) for n in handle.nodes if n.instance_id in by_id]
            if not adopted:
                raise RuntimeError("no live instances match the persisted handle")
            pool = session.adopt(
                name=handle.name,
                provider_config=provider_config,
                cluster=cluster,
                instances=tuple(inst for _, inst in adopted),
                node_ids=tuple(nid for nid, _ in adopted),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Reattach failed for {n}: {e}; cleaning up", n=handle.name, e=repr(exc),
            )
            _cleanup_dead(state, handle)
            continue
        state.register_adopted(handle.name, pool)
        subscribe_repersist(state, handle.name, provider_config, pool)
        logger.info("Reattached pool {n} ({k} nodes)", n=handle.name, k=len(handle.nodes))


def _cleanup_dead(state: ServerState, handle: SessionHandle) -> None:
    """Tear down remnant instances of a failed reattach and drop the handle."""
    with suppress(Exception):
        provider_config, cluster = unpack_payload(handle.payload)
        state.session.discard(provider_config=provider_config, cluster=cluster)
    remove_handle(handle.name)
