"""End-to-end reconciliation test through the real Casty client/worker stack.

Validates the full path that the node actor exercises on
``ConnectionRestored``: spawn a worker on a real ``ClusteredActorSystem``,
connect with a fresh ``ClusterClient`` (no SSH tunnel — direct loopback
TCP), discover the worker, dispatch a task with a stable ``task_id``, and
recover its result via ``GetResult`` even when the original ``ExecuteTask``
reply arrives before / during / after the reconcile probe.

These tests prove that:

1. The serializer wires ``GetResult`` / ``ResultPending`` / ``ResultDone``
   / ``ResultUnknown`` across a TCP boundary.
2. The cache can be queried mid-flight (``ResultPending``) and after
   completion (``ResultDone``) by an external client.
3. A fresh client on a fresh peer connection can hit ``GetResult`` on its
   very first ask — i.e. the reconcile probe doesn't depend on prior
   warm-up.  This is the failure mode flagged in the plan: if the peer
   is being re-established, the first ask must still land.
"""
from __future__ import annotations

import asyncio

import pytest
from casty import Behaviors, ClusterClient, ClusteredActorSystem

from skyward.infra.worker import (
    ExecuteTask,
    GetResult,
    ResultDone,
    ResultPending,
    ResultUnknown,
    TaskSucceeded,
    WORKER_KEY,
    skyward_serializer,
    worker_behavior,
)


pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


@pytest.fixture
async def worker_system():
    """A 1-node ClusteredActorSystem with a discoverable worker on a random port."""
    async with ClusteredActorSystem(
        name="skyward",
        host="127.0.0.1",
        port=0,
        node_id="node-0",
        seed_nodes=None,
        bind_host="127.0.0.1",
        serializer=skyward_serializer(),
    ) as system:
        worker_ref = system.spawn(
            Behaviors.discoverable(
                worker_behavior(node_id=0, concurrency=2),
                key=WORKER_KEY,
            ),
            "worker",
        )
        # surface the bound port so the client knows where to dial
        port = system.self_node.port
        yield system, worker_ref, port


@pytest.fixture
async def client_for(worker_system):
    """A fresh ClusterClient pointed at the worker_system fixture."""
    _, _, port = worker_system
    client = ClusterClient(
        contact_points=[("127.0.0.1", port)],
        system_name="skyward",
        serializer=skyward_serializer(),
    )
    await client.__aenter__()
    try:
        yield client
    finally:
        await client.__aexit__(None, None, None)


def _slow_compute(seconds: float) -> int:
    """A pickleable task that sleeps then returns 42."""
    import time
    time.sleep(seconds)
    return 42


async def _discover_worker(client: ClusterClient, timeout: float = 5.0):
    """Poll the receptionist until the worker is registered."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        listing = client.lookup(WORKER_KEY)
        if listing.instances:
            return next(iter(listing.instances)).ref
        await asyncio.sleep(0.1)
    raise TimeoutError("worker never registered")


class TestReconcileViaClient:
    @pytest.mark.asyncio
    async def test_pending_then_done_via_client(self, worker_system, client_for) -> None:
        """A real client can observe Pending mid-flight, then Done after completion."""
        import time as _time

        worker_ref = await _discover_worker(client_for)

        # dispatch a task that blocks the worker thread for ~600ms; do NOT await it.
        exec_task = asyncio.create_task(
            client_for.ask(
                worker_ref,
                lambda rto: ExecuteTask(
                    fn=_slow_compute, args=(0.6,), kwargs={},
                    reply_to=rto, task_id="rec-1",
                ),
                timeout=10.0,
            ),
        )
        # yield a few times so the create_task actually starts and the
        # ExecuteTask frame reaches the worker
        for _ in range(20):
            await asyncio.sleep(0.01)
            if exec_task.done():
                break

        # Probe mid-flight — task is still sleeping in the worker thread.
        pending = await client_for.ask(
            worker_ref,
            lambda rto: GetResult(task_id="rec-1", reply_to=rto),
            timeout=2.0,
        )
        assert isinstance(pending, ResultPending), (
            f"expected ResultPending while task sleeps, got {pending!r}"
        )

        # Wait for the original ask to complete; result delivered via the
        # original reply_to, not via reconciliation.
        result = await asyncio.wait_for(exec_task, timeout=5.0)
        assert isinstance(result, TaskSucceeded)
        assert result.result == 42

        # After completion the cache holds the result.
        done = await client_for.ask(
            worker_ref,
            lambda rto: GetResult(task_id="rec-1", reply_to=rto),
            timeout=2.0,
        )
        assert isinstance(done, ResultDone)
        assert isinstance(done.result, TaskSucceeded)
        assert done.result.result == 42

    @pytest.mark.asyncio
    async def test_cold_client_first_ask_recovers_result(self, worker_system) -> None:
        """A brand-new client (cold peers) can recover a cached result on its first ask.

        Validates the worst-case timing for the node actor's reconcile
        flow: the SSH tunnel just came back, ``ConnectionRestored`` fires,
        and the client immediately fans out ``GetResult``. There is no
        warm-up traffic between the client and worker — the reconcile
        probe IS the first thing on the new peer.
        """
        system, _worker_in_system, port = worker_system

        # Phase 1: warm client dispatches the task and gets the result.
        warm = ClusterClient(
            contact_points=[("127.0.0.1", port)],
            system_name="skyward",
            serializer=skyward_serializer(),
        )
        await warm.__aenter__()
        try:
            worker_ref_warm = await _discover_worker(warm)
            result = await warm.ask(
                worker_ref_warm,
                lambda rto: ExecuteTask(
                    fn=lambda: 99, args=(), kwargs={},
                    reply_to=rto, task_id="rec-2",
                ),
                timeout=5.0,
            )
            assert isinstance(result, TaskSucceeded)
            assert result.result == 99
        finally:
            await warm.__aexit__(None, None, None)

        # Phase 2: simulate a fresh client coming up after a tunnel
        # reconnect. Its very first ask is GetResult — exactly what the
        # node actor does on ConnectionRestored.
        cold = ClusterClient(
            contact_points=[("127.0.0.1", port)],
            system_name="skyward",
            serializer=skyward_serializer(),
        )
        await cold.__aenter__()
        try:
            worker_ref_cold = await _discover_worker(cold)
            done = await cold.ask(
                worker_ref_cold,
                lambda rto: GetResult(task_id="rec-2", reply_to=rto),
                timeout=5.0,
            )
            assert isinstance(done, ResultDone), (
                f"expected ResultDone on cold client's first ask, got {done!r}"
            )
            assert isinstance(done.result, TaskSucceeded)
            assert done.result.result == 99
        finally:
            await cold.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_unknown_task_id_via_client(self, worker_system, client_for) -> None:
        """Asking for a task_id the worker never saw returns ResultUnknown."""
        worker_ref = await _discover_worker(client_for)
        reply = await client_for.ask(
            worker_ref,
            lambda rto: GetResult(task_id="never-existed", reply_to=rto),
            timeout=2.0,
        )
        assert isinstance(reply, ResultUnknown)
