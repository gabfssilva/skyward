"""Integration test: ClusterClient routing through a real SSH forward tunnel.

Starts an in-process asyncssh server, creates a forward tunnel from a local
port to the cluster's TCP port, and verifies that ClusterClient can route
messages end-to-end through the tunnel.

Tests both patterns:
- Sharded entities (entity_ref + ShardEnvelope)
- Discoverable actors (lookup + direct ask)
"""
from __future__ import annotations

import asyncio
import os
import tempfile
from dataclasses import dataclass
from typing import Any

import asyncssh
import pytest

from casty import ActorContext, ActorRef, Behavior, Behaviors, ServiceKey
from casty import ClusterClient
from casty import ClusteredActorSystem, ShardEnvelope


@dataclass(frozen=True)
class Ping:
    reply_to: ActorRef[str]


type EchoMsg = Ping


@dataclass(frozen=True)
class WorkerPing:
    payload: str
    reply_to: ActorRef[str]


type WorkerMsg = WorkerPing

WORKER_KEY: ServiceKey[WorkerMsg] = ServiceKey("test-worker")


def discoverable_worker(node_id: str) -> Behavior[WorkerMsg]:
    async def receive(_ctx: ActorContext[WorkerMsg], msg: WorkerMsg) -> Behavior[WorkerMsg]:
        match msg:
            case WorkerPing(payload=payload, reply_to=reply_to):
                reply_to.tell(f"pong-{node_id}-{payload}")
                return Behaviors.same()
            case _:
                return Behaviors.same()

    return Behaviors.receive(receive)


def echo_entity(entity_id: str) -> Behavior[EchoMsg]:
    async def receive(_ctx: Any, msg: Any) -> Any:
        match msg:
            case Ping(reply_to=reply_to):
                reply_to.tell(f"pong-{entity_id}")
                return Behaviors.same()
            case _:
                return Behaviors.same()

    return Behaviors.receive(receive)


class TunnelSSHServer(asyncssh.SSHServer):
    """Minimal SSH server that allows TCP forwarding."""

    def begin_auth(self, username: str) -> bool:
        return False

    def connection_requested(
        self, dest_host: str, dest_port: int, orig_host: str, orig_port: int
    ) -> bool:
        return True


@pytest.mark.timeout(30)
async def test_client_through_ssh_forward_tunnel() -> None:
    """Messages flow through a real SSH forward tunnel to a cluster node."""
    key = asyncssh.generate_private_key("ssh-rsa")
    with tempfile.NamedTemporaryFile(suffix=".key", delete=False) as f:
        f.write(key.export_private_key())
        key_file = f.name

    try:
        # 1. Start SSH server
        ssh_server = await asyncssh.create_server(
            TunnelSSHServer,
            "127.0.0.1",
            0,
            server_host_keys=[key_file],
        )
        ssh_port = ssh_server.sockets[0].getsockname()[1]

        # 2. Start cluster node
        async with ClusteredActorSystem(
            name="tunnel-cluster",
            host="127.0.0.1",
            port=0,
            node_id="node-1",
        ) as system:
            cluster_port = system.self_node.port

            system.spawn(
                Behaviors.sharded(entity_factory=echo_entity, num_shards=10),
                "echo",
            )
            await asyncio.sleep(0.3)

            # 3. Create forward tunnel: local_port -> SSH -> cluster_port
            async with asyncssh.connect(
                "127.0.0.1",
                ssh_port,
                known_hosts=None,
            ) as ssh_conn:
                listener = await ssh_conn.forward_local_port(
                    "", 0, "127.0.0.1", cluster_port
                )
                tunnel_port = listener.get_port()

                # 4. ClusterClient routing through the tunnel
                async with ClusterClient(
                    contact_points=[("10.99.99.1", 25520)],
                    system_name="tunnel-cluster",
                    address_map={
                        ("10.99.99.1", 25520): ("127.0.0.1", tunnel_port),
                        ("127.0.0.1", cluster_port): (
                            "127.0.0.1",
                            tunnel_port,
                        ),
                    },
                ) as client:
                    await asyncio.sleep(1.5)

                    echo_ref = client.entity_ref("echo", num_shards=10)
                    await asyncio.sleep(0.5)
                    result = await client.ask(
                        echo_ref,
                        lambda r: ShardEnvelope(
                            "test-entity", Ping(reply_to=r)
                        ),
                        timeout=5.0,
                    )
                    assert result == "pong-test-entity"

        ssh_server.close()
    finally:
        os.unlink(key_file)


@pytest.mark.timeout(30)
async def test_client_multiple_asks_through_tunnel() -> None:
    """Multiple ask() calls succeed through the SSH tunnel."""
    key = asyncssh.generate_private_key("ssh-rsa")
    with tempfile.NamedTemporaryFile(suffix=".key", delete=False) as f:
        f.write(key.export_private_key())
        key_file = f.name

    try:
        ssh_server = await asyncssh.create_server(
            TunnelSSHServer,
            "127.0.0.1",
            0,
            server_host_keys=[key_file],
        )
        ssh_port = ssh_server.sockets[0].getsockname()[1]

        async with ClusteredActorSystem(
            name="tunnel-cluster",
            host="127.0.0.1",
            port=0,
            node_id="node-1",
        ) as system:
            cluster_port = system.self_node.port

            system.spawn(
                Behaviors.sharded(entity_factory=echo_entity, num_shards=10),
                "echo",
            )
            await asyncio.sleep(0.3)

            async with asyncssh.connect(
                "127.0.0.1", ssh_port, known_hosts=None,
            ) as ssh_conn:
                listener = await ssh_conn.forward_local_port(
                    "", 0, "127.0.0.1", cluster_port
                )
                tunnel_port = listener.get_port()

                async with ClusterClient(
                    contact_points=[("10.99.99.1", 25520)],
                    system_name="tunnel-cluster",
                    address_map={
                        ("10.99.99.1", 25520): ("127.0.0.1", tunnel_port),
                        ("127.0.0.1", cluster_port): (
                            "127.0.0.1",
                            tunnel_port,
                        ),
                    },
                ) as client:
                    await asyncio.sleep(1.5)

                    echo_ref = client.entity_ref("echo", num_shards=10)
                    await asyncio.sleep(0.5)

                    for entity_id in ("alice", "bob", "carol"):
                        eid = entity_id
                        result = await client.ask(
                            echo_ref,
                            lambda r, e=eid: ShardEnvelope(
                                e, Ping(reply_to=r)
                            ),
                            timeout=5.0,
                        )
                        assert result == f"pong-{entity_id}"

        ssh_server.close()
    finally:
        os.unlink(key_file)


@pytest.mark.timeout(30)
async def test_discoverable_lookup_ask_through_tunnel() -> None:
    """Discoverable actors found via lookup() and asked through SSH tunnel."""
    key = asyncssh.generate_private_key("ssh-rsa")
    with tempfile.NamedTemporaryFile(suffix=".key", delete=False) as f:
        f.write(key.export_private_key())
        key_file = f.name

    try:
        ssh_server = await asyncssh.create_server(
            TunnelSSHServer,
            "127.0.0.1",
            0,
            server_host_keys=[key_file],
        )
        ssh_port = ssh_server.sockets[0].getsockname()[1]

        async with ClusteredActorSystem(
            name="tunnel-cluster",
            host="127.0.0.1",
            port=0,
            node_id="node-1",
        ) as system:
            cluster_port = system.self_node.port

            system.spawn(
                Behaviors.discoverable(
                    discoverable_worker("node-1"),
                    key=WORKER_KEY,
                ),
                "worker",
            )
            await asyncio.sleep(0.3)

            async with asyncssh.connect(
                "127.0.0.1", ssh_port, known_hosts=None,
            ) as ssh_conn:
                listener = await ssh_conn.forward_local_port(
                    "", 0, "127.0.0.1", cluster_port
                )
                tunnel_port = listener.get_port()

                async with ClusterClient(
                    contact_points=[("10.99.99.1", 25520)],
                    system_name="tunnel-cluster",
                    address_map={
                        ("10.99.99.1", 25520): ("127.0.0.1", tunnel_port),
                        ("127.0.0.1", cluster_port): (
                            "127.0.0.1",
                            tunnel_port,
                        ),
                    },
                ) as client:
                    await asyncio.sleep(1.5)

                    listing = client.lookup(WORKER_KEY)
                    instances = list(listing.instances)
                    assert len(instances) >= 1, (
                        f"Expected 1 worker, found {len(instances)}"
                    )

                    worker_ref = instances[0].ref
                    result = await client.ask(
                        worker_ref,
                        lambda r: WorkerPing(payload="hello", reply_to=r),
                        timeout=5.0,
                    )
                    assert result == "pong-node-1-hello"

        ssh_server.close()
    finally:
        os.unlink(key_file)


@pytest.mark.timeout(30)
async def test_discoverable_ask_from_actor_through_tunnel() -> None:
    """client.ask from inside an actor behavior through SSH tunnel.

    Replicates the executor pattern: an actor spawned in a local ActorSystem
    uses ClusterClient.ask to reach a discoverable worker on a remote cluster
    through an SSH forward tunnel.
    """

    @dataclass(frozen=True)
    class DoAsk:
        reply_to: ActorRef[str]

    @dataclass(frozen=True)
    class _AskDone:
        result: str
        reply_to: ActorRef[str]

    type _AskMsg = DoAsk | _AskDone

    key = asyncssh.generate_private_key("ssh-rsa")
    with tempfile.NamedTemporaryFile(suffix=".key", delete=False) as f:
        f.write(key.export_private_key())
        key_file = f.name

    try:
        ssh_server = await asyncssh.create_server(
            TunnelSSHServer,
            "127.0.0.1",
            0,
            server_host_keys=[key_file],
        )
        ssh_port = ssh_server.sockets[0].getsockname()[1]

        async with ClusteredActorSystem(
            name="tunnel-cluster",
            host="127.0.0.1",
            port=0,
            node_id="node-1",
        ) as remote_system:
            cluster_port = remote_system.self_node.port

            remote_system.spawn(
                Behaviors.discoverable(
                    discoverable_worker("node-1"),
                    key=WORKER_KEY,
                ),
                "worker",
            )
            await asyncio.sleep(0.3)

            async with asyncssh.connect(
                "127.0.0.1", ssh_port, known_hosts=None,
            ) as ssh_conn:
                listener = await ssh_conn.forward_local_port(
                    "", 0, "127.0.0.1", cluster_port
                )
                tunnel_port = listener.get_port()

                async with ClusterClient(
                    contact_points=[("10.99.99.1", 25520)],
                    system_name="tunnel-cluster",
                    address_map={
                        ("10.99.99.1", 25520): ("127.0.0.1", tunnel_port),
                        ("127.0.0.1", cluster_port): (
                            "127.0.0.1",
                            tunnel_port,
                        ),
                    },
                ) as client:
                    await asyncio.sleep(1.5)

                    listing = client.lookup(WORKER_KEY)
                    instances = list(listing.instances)
                    assert len(instances) >= 1

                    worker_ref = instances[0].ref

                    from casty import ActorSystem

                    def asker_behavior(
                        cl: ClusterClient, wr: ActorRef[Any],
                    ) -> Behavior[_AskMsg]:
                        async def receive(
                            ctx: ActorContext[_AskMsg], msg: _AskMsg,
                        ) -> Behavior[_AskMsg]:
                            match msg:
                                case DoAsk(reply_to=reply_to):
                                    ctx.pipe_to_self(
                                        coro=cl.ask(
                                            wr,
                                            lambda r: WorkerPing(
                                                payload="from-actor",
                                                reply_to=r,
                                            ),
                                            timeout=5.0,
                                        ),
                                        mapper=lambda result: _AskDone(
                                            result=result,
                                            reply_to=reply_to,
                                        ),
                                        on_failure=lambda e: _AskDone(
                                            result=f"error: {e}",
                                            reply_to=reply_to,
                                        ),
                                    )
                                    return Behaviors.same()
                                case _AskDone(
                                    result=result, reply_to=reply_to,
                                ):
                                    reply_to.tell(result)
                                    return Behaviors.same()

                        return Behaviors.receive(receive)

                    async with ActorSystem("local-executor") as local_system:
                        asker_ref = local_system.spawn(
                            asker_behavior(client, worker_ref), "asker",
                        )
                        result = await local_system.ask(
                            asker_ref,
                            lambda r: DoAsk(reply_to=r),
                            timeout=10.0,
                        )
                        assert result == "pong-node-1-from-actor"

        ssh_server.close()
    finally:
        os.unlink(key_file)


@pytest.mark.timeout(30)
async def test_multinode_per_node_tunnels() -> None:
    """Two-node cluster with separate SSH tunnel per node.

    Each node gets its own SSH forward tunnel. The ClusterClient address_map
    routes each node's logical address to its dedicated tunnel endpoint.
    """
    key = asyncssh.generate_private_key("ssh-rsa")
    with tempfile.NamedTemporaryFile(suffix=".key", delete=False) as f:
        f.write(key.export_private_key())
        key_file = f.name

    try:
        ssh_server = await asyncssh.create_server(
            TunnelSSHServer,
            "127.0.0.1",
            0,
            server_host_keys=[key_file],
        )
        ssh_port = ssh_server.sockets[0].getsockname()[1]

        async with ClusteredActorSystem(
            name="tunnel-cluster",
            host="127.0.0.1",
            port=0,
            node_id="node-0",
        ) as system_0:
            port_0 = system_0.self_node.port

            async with ClusteredActorSystem(
                name="tunnel-cluster",
                host="127.0.0.1",
                port=0,
                node_id="node-1",
                seed_nodes=[("127.0.0.1", port_0)],
            ) as system_1:
                port_1 = system_1.self_node.port

                system_0.spawn(
                    Behaviors.discoverable(
                        discoverable_worker("node-0"), key=WORKER_KEY,
                    ),
                    "worker",
                )
                system_1.spawn(
                    Behaviors.discoverable(
                        discoverable_worker("node-1"), key=WORKER_KEY,
                    ),
                    "worker",
                )
                await asyncio.sleep(1.0)

                async with asyncssh.connect(
                    "127.0.0.1", ssh_port, known_hosts=None,
                ) as ssh_0, asyncssh.connect(
                    "127.0.0.1", ssh_port, known_hosts=None,
                ) as ssh_1:
                    listener_0 = await ssh_0.forward_local_port(
                        "", 0, "127.0.0.1", port_0,
                    )
                    listener_1 = await ssh_1.forward_local_port(
                        "", 0, "127.0.0.1", port_1,
                    )
                    tport_0 = listener_0.get_port()
                    tport_1 = listener_1.get_port()

                    address_map = {
                        ("10.0.0.1", 25520): ("127.0.0.1", tport_0),
                        ("127.0.0.1", port_0): ("127.0.0.1", tport_0),
                        ("127.0.0.1", port_1): ("127.0.0.1", tport_1),
                    }

                    async with ClusterClient(
                        contact_points=[("10.0.0.1", 25520)],
                        system_name="tunnel-cluster",
                        address_map=address_map,
                    ) as client:
                        await asyncio.sleep(2.0)

                        listing = client.lookup(WORKER_KEY)
                        instances = list(listing.instances)
                        assert len(instances) >= 2, (
                            f"Expected 2 workers, found {len(instances)}"
                        )

                        host_to_node: dict[str, str] = {}
                        for inst in instances:
                            host_to_node[inst.node.host] = inst.node.host

                        results: set[str] = set()
                        for inst in instances:
                            result = await client.ask(
                                inst.ref,
                                lambda r: WorkerPing(
                                    payload="multi", reply_to=r,
                                ),
                                timeout=5.0,
                            )
                            results.add(result)

                        assert results == {
                            "pong-node-0-multi",
                            "pong-node-1-multi",
                        }

        ssh_server.close()
    finally:
        os.unlink(key_file)
