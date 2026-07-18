"""A local Jupyter, a kernel on a Skyward machine.

The provisioner runs inside the Jupyter process and owns two things: a task on
the compute that is the kernel, and five loopback ports that reach it. The task
is a streaming call — it starts ``ipykernel_launcher`` on the machine, hands
back the ports and HMAC key it chose, and then blocks on the process, so the
stream is alive for exactly as long as the kernel is. Closing the stream is what
kills it.

Nothing here opens an SSH connection. v1 dialed the node itself; v2 has no node
to dial — the client speaks HTTP to the daemon and the daemon owns the machines
— so the ZMQ channels ride the daemon's port forwarding, the same bridge
``sky.Port`` uses.
"""

from __future__ import annotations

import asyncio
import signal
from collections.abc import Callable, Generator, Iterator
from contextlib import ExitStack, suppress
from typing import Any

from jupyter_client.connect import KernelConnectionInfo
from jupyter_client.provisioning.provisioner_base import KernelProvisionerBase
from traitlets import Unicode

from skyward.sdk.compute import Compute
from skyward.sdk.forward import TcpProxy
from skyward.sdk.function import Streaming
from skyward.sdk.spec import Port

__all__ = ["SkywardKernelProvisioner"]

CHANNELS = ("shell_port", "iopub_port", "stdin_port", "control_port", "hb_port")

type Connection = dict[str, str | int]


def _kernel() -> Callable[[], Iterator[Connection]]:
    """The generator that runs on the machine, built where it cannot be imported.

    It is nested so that it is pickled by value: importing this module on a node
    would pull in ``jupyter_client``, which is a client-side extra and is not
    there. Nothing it needs comes from this module's namespace.
    """

    def kernel() -> Iterator[Connection]:
        import importlib.util
        import json
        import os
        import secrets
        import socket
        import subprocess
        import sys
        import tempfile
        import time
        from contextlib import suppress

        if importlib.util.find_spec("ipykernel") is None:
            raise RuntimeError("the machine has no ipykernel; give the compute an image with it: sky.Image(pip=['ipykernel'])")

        def free() -> int:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                return probe.getsockname()[1]

        def listening(port: int) -> bool:
            with socket.socket() as probe:
                return probe.connect_ex(("127.0.0.1", port)) == 0

        channels = ("shell_port", "iopub_port", "stdin_port", "control_port", "hb_port")
        connection: dict[str, str | int] = {
            "ip": "127.0.0.1",
            "transport": "tcp",
            "signature_scheme": "hmac-sha256",
            "key": secrets.token_hex(16),
            "kernel_name": "python3",
        } | {channel: free() for channel in channels}

        path = os.path.join(tempfile.gettempdir(), f"skyward-kernel-{secrets.token_hex(8)}.json")
        with open(path, "w") as file:
            json.dump(connection, file)

        process = subprocess.Popen([sys.executable, "-m", "ipykernel_launcher", "-f", path])
        try:
            deadline = time.monotonic() + 60
            while not all(listening(int(connection[channel])) for channel in channels):
                if process.poll() is not None:
                    raise RuntimeError(f"the kernel exited before it bound its channels (status {process.returncode})")
                if time.monotonic() > deadline:
                    raise RuntimeError("the kernel did not bind its channels within 60s")
                time.sleep(0.1)

            yield connection
            process.wait()
        finally:
            with suppress(Exception):
                process.kill()
            with suppress(OSError):
                os.remove(path)

    return kernel


class SkywardKernelProvisioner(KernelProvisionerBase):
    """Run a Jupyter kernel on a Skyward compute, reached through the daemon."""

    compute = Unicode(config=True)
    url = Unicode("", config=True)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._stack = ExitStack()
        self._pool: Compute | None = None
        self._stream: Generator[Connection, None, None] | None = None
        self._proxies: list[TcpProxy] = []
        self._connection: Connection = {}
        self._local: dict[str, int] = {}

    @property
    def has_process(self) -> bool:
        """Whether a kernel is running on the machine."""
        return self._stream is not None

    async def pre_launch(self, **kwargs: Any) -> dict[str, Any]:
        """Attach to the compute, start the kernel on it, and bridge its channels."""
        kwargs = await super().pre_launch(**kwargs)
        await asyncio.to_thread(self._attach)
        await asyncio.to_thread(self._start)
        await asyncio.to_thread(self._forward)
        kwargs["cmd"] = ["python", "-m", "ipykernel_launcher"]
        return kwargs

    async def launch_kernel(self, cmd: list[str], **kwargs: Any) -> KernelConnectionInfo:
        """Return the local connection info. The kernel is already up; ``cmd`` is inert."""
        info: KernelConnectionInfo = {
            "ip": "127.0.0.1",
            "transport": "tcp",
            "signature_scheme": "hmac-sha256",
            "key": str(self._connection["key"]),
            "shell_port": self._local["shell_port"],
            "iopub_port": self._local["iopub_port"],
            "stdin_port": self._local["stdin_port"],
            "control_port": self._local["control_port"],
            "hb_port": self._local["hb_port"],
        }
        self.connection_info = info
        return info

    async def poll(self) -> int | None:
        """None while the kernel task is alive."""
        return None if self._stream is not None else 0

    async def wait(self) -> int | None:
        """Block until the kernel process on the machine exits.

        The remote generator is parked on ``process.wait()`` after its one frame,
        so asking it for another is asking when the kernel died.
        """
        if self._stream is None:
            return 0
        await asyncio.to_thread(next, self._stream, None)
        self._stream = None
        return 0

    async def send_signal(self, signum: int) -> None:
        """Terminate on SIGTERM/SIGKILL; ignore the rest.

        There is no signal path to a process the daemon owns. Interrupts do not
        need one — the kernelspec asks for ``interrupt_mode: message``, so they
        go down the control channel like any other request.
        """
        if signum in (signal.SIGTERM, signal.SIGKILL):
            await self.terminate()

    async def terminate(self, restart: bool = False) -> None:
        """End the kernel by ending its stream."""
        await asyncio.to_thread(self._stop)

    async def kill(self, restart: bool = False) -> None:
        """Same as terminate: the stream is the only handle on the process."""
        await asyncio.to_thread(self._stop)

    async def cleanup(self, restart: bool = False) -> None:
        """Close the port bridges and the kernel; detach from the compute unless restarting."""
        await asyncio.to_thread(self._cleanup, restart)

    def _attach(self) -> None:
        """Join the compute the kernelspec names, and refuse a shape that cannot work.

        Both the streaming task and the forwarded connections are routed by the
        daemon, independently, round-robin. On one ready node those agree; on more
        than one they do not, and the kernel would be on a different machine from
        the ports reaching it.
        """
        pool = self._stack.enter_context(Compute.attached(self.compute, url=self.url or None, console=False))
        if (ready := pool.current_nodes()) != 1:
            raise RuntimeError(f"the remote kernel needs a compute with exactly one ready node, and {self.compute!r} has {ready}")
        self._pool = pool

    def _start(self) -> None:
        self._stream = self._frames()
        self._connection = next(self._stream)

    def _frames(self) -> Generator[Connection, None, None]:
        """The pool's stream, as a generator this can close.

        ``Compute.stream`` promises an iterator, and closing is the whole handle
        on the kernel: ``yield from`` is what carries the close through to it.
        """
        yield from self.pool.stream(Streaming(_kernel(), (), {}))

    def _forward(self) -> None:
        for channel in CHANNELS:
            proxy = TcpProxy(self.pool.client, self.pool.id, Port(remote=int(self._connection[channel])))
            self.pool.loop.run(proxy.start())
            self._proxies.append(proxy)
            self._local[channel] = proxy.local

    def _stop(self) -> None:
        if self._stream is not None:
            with suppress(Exception):
                self._stream.close()
            self._stream = None

    def _cleanup(self, restart: bool) -> None:
        for proxy in self._proxies:
            with suppress(Exception):
                self.pool.loop.run(proxy.stop())
        self._proxies = []
        self._local = {}
        self._stop()
        if not restart:
            self._stack.close()
            self._pool = None

    @property
    def pool(self) -> Compute:
        if self._pool is None:
            raise RuntimeError("the kernel is not attached to a compute")
        return self._pool
