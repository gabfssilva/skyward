"""Shared RPC execution logic for all providers."""

from __future__ import annotations

import contextvars
import time
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import cloudpickle  # type: ignore[import-untyped]
import rpyc  # type: ignore[import-untyped]

from skyward.callback import emit
from skyward.events import LogLine
from skyward.exceptions import ConnectionLostError
from skyward.providers.common.accelerator_detection import resolve_accelerator
from skyward.providers.common.types import (
    HeadAddrResolver,
    PeerResolver,
)
from skyward.providers.pool_info import build_pool_info
from skyward.serialization import deserialize, serialize

if TYPE_CHECKING:
    from skyward.types import Instance


def safe_rpc_call[T](instance_id: str, fn: Callable[[], T]) -> T:
    """Execute RPyC call with connection error handling."""
    try:
        return fn()
    except (EOFError, ConnectionResetError, BrokenPipeError, OSError) as e:
        raise ConnectionLostError(instance_id, str(e)) from e


def run[R](
    conns: dict[str, rpyc.Connection],
    instances: tuple[Instance, ...],
    compute: Any,  # _PoolCompute in v2
    args: tuple[object, ...],
    kwargs: dict[str, object],
    *,
    resolve_peers: PeerResolver,
    resolve_head_addr: HeadAddrResolver,
) -> list[R]:
    """Execute function on instances via RPyC.

    This is the shared execution engine. Provider-specific behavior is
    injected via the resolver functions. Events are emitted via the
    callback system (see skyward.callback).

    Args:
        conns: Dict mapping instance_id to rpyc.Connection.
        instances: Tuple of target instances.
        compute: Compute specification.
        args: Function arguments.
        kwargs: Function keyword arguments.
        resolve_peers: Function to build peer info from instances.
        resolve_head_addr: Function to get head node address.

    Returns:
        List of results from each node.
    """
    cluster_id = instances[0].get_meta("cluster_id", str(uuid.uuid4())[:8])
    peers = resolve_peers(instances)
    head_addr = resolve_head_addr(instances)

    # Convert PeerInfo to dicts for build_pool_info
    peers_dicts = [
        {"node": p.node, "private_ip": p.private_ip, "addr": p.addr}
        for p in peers
    ]

    # Setup cluster environment on all nodes
    for node, instance in enumerate(instances):
        accelerator_type, accelerator_count = resolve_accelerator(instance)

        pool_info = build_pool_info(
            node=node,
            total_nodes=len(instances),
            accelerator_count=accelerator_count,
            total_accelerators=len(instances) * accelerator_count,
            head_addr=head_addr,
            head_port=compute.head_port,
            job_id=cluster_id,
            peers=peers_dicts,
            accelerator_type=accelerator_type,
        )

        env_bytes = cloudpickle.dumps(compute.env_dict)

        def _setup_cluster(
            i: Instance = instance, p: Any = pool_info, e: bytes = env_bytes
        ) -> None:
            conns[i.id].root.setup_cluster(p.model_dump_json(), e)

        safe_rpc_call(instance.id, _setup_cluster)

    # Serialize function and arguments
    fn_bytes = serialize(compute.wrapped_fn)
    args_bytes = serialize(args)
    kwargs_bytes = serialize(kwargs)

    # Execute on all nodes in parallel
    from skyward.conc import map_async_indexed

    # Capture context BEFORE spawning threads (main thread has _callback set)
    ctx = contextvars.copy_context()

    def execute_node(node: int, instance: Instance) -> R:
        conn = conns[instance.id]

        def stdout_cb(line: str) -> None:
            ctx.run(
                emit,
                LogLine(
                    node=node,
                    instance_id=instance.id,
                    line=line,
                    timestamp=time.time(),
                ),
            )

        try:
            result_bytes = safe_rpc_call(
                instance.id,
                lambda: conn.root.execute(fn_bytes, args_bytes, kwargs_bytes, stdout_cb),
            )
            data = deserialize(result_bytes)

            if data.get("error"):
                raise RuntimeError(f"Node {node} failed: {data['error']}")

            return cast(R, data["result"])
        except Exception as e:
            raise RuntimeError(f"Node {node} ({instance.id}) failed: {e}") from e

    return list(map_async_indexed(execute_node, instances))
