"""End-to-end reconciliation test through the real casty client/worker stack.

Validates the full path the node exercises after a dropped ``run`` RPC:
start a real casty node hosting the worker services, connect with a lite
``casty.connect`` client (no SSH tunnel — direct loopback TCP), pin the
services to the member, dispatch a task with a stable ``task_id``, and
recover its result via ``WorkerControl.get_result`` even when the original
``run`` reply arrives before / during / after the reconcile probe.

These tests prove that:

1. The payload codec wires ``get_result`` replies across a TCP boundary.
2. The cache can be queried mid-flight (``ResultPending``) and after
   completion (``ResultDone``) by an external client.
3. A fresh client on a fresh peer connection can hit ``get_result`` on its
   very first ask — the reconcile probe doesn't depend on prior warm-up.
"""
from __future__ import annotations

import asyncio
import socket

import casty
import pytest

import skyward.infra.worker as worker_mod
from skyward.infra.worker import (
    ResultDone,
    ResultPending,
    ResultUnknown,
    TaskSucceeded,
    WorkerControl,
    WorkerService,
    _Runtime,
    dumps,
    loads,
)

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
async def worker_system():
    """A 1-node casty cluster hosting the worker services on a random port."""
    port = _free_port()
    async with casty.start(f"127.0.0.1:{port}", cluster_name="skyward") as node:
        worker_mod._runtime = _Runtime(
            node_id=0, loop=asyncio.get_running_loop(),
        )
        yield node, port
        worker_mod._runtime = None


async def _connect(port: int) -> casty.Client:
    return await casty.connect(
        seeds=[f"127.0.0.1:{port}"], cluster_name="skyward",
    )


async def _pinned(client: casty.Client, timeout: float = 5.0):
    """Wait for the member and return (service, control) pinned to it."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        members = client.members()
        if members:
            member = next(iter(members))
            return (
                client.service(WorkerService, at=member),
                client.service(WorkerControl, at=member),
            )
        await asyncio.sleep(0.1)
    raise TimeoutError("member never appeared")


def _slow_compute(seconds: float) -> int:
    """A pickleable task that sleeps then returns 42."""
    import time
    time.sleep(seconds)
    return 42


def _payload(fn, args=()) -> bytes:
    return dumps((fn, args, {}, ()))


class TestReconcileViaClient:
    async def test_pending_then_done_via_client(self, worker_system) -> None:
        """A real client can observe Pending mid-flight, then Done after completion."""
        _, port = worker_system
        client = await _connect(port)
        try:
            service, control = await _pinned(client)

            exec_task = asyncio.create_task(
                service.run("rec-1", _payload(_slow_compute, (0.6,))),
            )
            for _ in range(20):
                await asyncio.sleep(0.01)
                if exec_task.done():
                    break

            pending = loads(await control.get_result("rec-1"))
            assert isinstance(pending, ResultPending), (
                f"expected ResultPending while task sleeps, got {pending!r}"
            )

            result = loads(await asyncio.wait_for(exec_task, timeout=5.0))
            assert isinstance(result, TaskSucceeded)
            assert result.result == 42

            done = loads(await control.get_result("rec-1"))
            assert isinstance(done, ResultDone)
            assert isinstance(done.result, TaskSucceeded)
            assert done.result.result == 42
        finally:
            await client.close()

    async def test_cold_client_first_ask_recovers_result(self, worker_system) -> None:
        """A brand-new client (cold peers) can recover a cached result on its first ask.

        Validates the worst-case timing for the node's reconcile flow: the
        SSH tunnel just came back and the client immediately calls
        ``get_result`` — no warm-up traffic on the new peer.
        """
        _, port = worker_system

        warm = await _connect(port)
        try:
            service, _ = await _pinned(warm)
            result = loads(await service.run("rec-2", _payload(_ninety_nine)))
            assert isinstance(result, TaskSucceeded)
            assert result.result == 99
        finally:
            await warm.close()

        cold = await _connect(port)
        try:
            _, control = await _pinned(cold)
            done = loads(await control.get_result("rec-2"))
            assert isinstance(done, ResultDone), (
                f"expected ResultDone on cold client's first ask, got {done!r}"
            )
            assert isinstance(done.result, TaskSucceeded)
            assert done.result.result == 99
        finally:
            await cold.close()

    async def test_unknown_task_id_via_client(self, worker_system) -> None:
        """Asking for a task_id the worker never saw returns ResultUnknown."""
        _, port = worker_system
        client = await _connect(port)
        try:
            _, control = await _pinned(client)
            reply = loads(await control.get_result("never-existed"))
            assert isinstance(reply, ResultUnknown)
        finally:
            await client.close()


def _ninety_nine() -> int:
    return 99
