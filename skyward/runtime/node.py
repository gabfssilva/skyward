from __future__ import annotations

import asyncio
import logging
import shlex
from collections.abc import Callable

import msgspec

from skyward.application.provider import Machine
from skyward.protocol.schemas import Image, NodeState, PluginRef
from skyward.runtime import bootstrap, worker
from skyward.runtime.events import events
from skyward.runtime.journal import SKYWARD_DIR, Console, NodeEvent, Phase
from skyward.runtime.source import Source
from skyward.runtime.ssh import SshChannel

logger = logging.getLogger(__name__)

BOOTSTRAP_TIMEOUT = 900.0
WORKER_TIMEOUT = 180.0

type Listener = Callable[[NodeState, str | None], None]
type Output = Callable[[str, str | None], None]


class BootstrapFailedError(RuntimeError):
    """The machine came up and could not be made into a worker."""


class Node:
    """One machine, from reachable to usable.

    A linear lifecycle written as a linear coroutine: connect, bootstrap, start
    the worker, be ready. It is the only part of the system where a sequence is
    honestly a sequence, and the only part that holds something a database cannot
    — a live connection and the tunnel through it.

    It decides nothing. A node that fails says so and stops; whether a lost node
    is worth replacing is a question about what the compute was asked for, and the
    reconciler is the one holding that. This is why there is no retry policy here:
    a dead node is a deficit, and a deficit is already something the reconciler
    knows how to close.
    """

    def __init__(
        self,
        machine: Machine,
        compute: str,
        private_key: str,
        image: Image,
        source: Source,
        listener: Listener,
        output: Output,
        rank: int = 0,
        peers: tuple[str, ...] = (),
        seeds: tuple[str, ...] = (),
        concurrency: int = 1,
        plugins: tuple[PluginRef, ...] = (),
    ) -> None:
        if machine.host is None:
            raise ValueError(f"machine {machine.id} has no address to connect to")

        self._machine = machine
        self._compute = compute
        self._image = image
        self._source = source
        self._listener = listener
        self._output = output
        self._rank = rank
        self._peers = peers
        self._seeds = seeds
        self._concurrency = concurrency
        self._plugins = plugins
        self._ssh = SshChannel(
            machine.host,
            port=machine.port,
            user=machine.user,
            private_key=private_key,
            password=machine.password,
        )
        self._lifecycle: asyncio.Task[None] | None = None
        self._monitor: asyncio.Task[None] | None = None
        self._reached: dict[str, asyncio.Future[str | None]] = {}
        self._failure: str | None = None
        self.tunnel: int | None = None
        """The local port the daemon's casty client dials to reach this worker."""

    async def start(self) -> None:
        self._lifecycle = asyncio.create_task(self._run())

    async def close(self) -> None:
        for task in (self._lifecycle, self._monitor):
            if task:
                task.cancel()
        await self._ssh.close()

    @property
    def peer(self) -> str:
        """Where the other machines reach this one."""
        return self._machine.private_host or self._machine.host or ""

    @property
    def seed(self) -> str:
        """Where this node would be reached, if it were the one others joined."""
        return f"{self.peer}:{worker.PORT}"

    async def _run(self) -> None:
        try:
            self._listener("connecting", None)
            await self._ssh.connect()
            self._monitor = asyncio.create_task(self._watch())

            if await self._serving():
                self.tunnel = await self._ssh.forward(worker.PORT)
                self._listener("ready", None)
                return

            self._listener("bootstrapping", None)
            await self._bootstrap()
            await self._launch()

            self._listener("ready", None)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("node %s failed: %s", self._machine.id, exc)
            self._listener("failed", str(exc))

    async def _serving(self) -> bool:
        """Whether this machine is already a worker.

        The machine outlives the process that made it one, so arriving at a machine
        that is already working is the normal case, not the exception: another
        process attached to the compute, or this daemon came back up. Bootstrapping
        it again would reinstall a venv that is fine and then start a second worker
        onto a port the first one holds.

        Adopting it is also what makes the tasks in flight survive: the worker never
        stopped, so neither did they.
        """
        result = await self._ssh.run("pgrep -f skyward.runtime.worker")
        return result.exit_code == 0

    async def _bootstrap(self) -> None:
        await self._ssh.run(f"mkdir -p {SKYWARD_DIR}")
        if self._source.wheel:
            await self._ssh.put(self._source.argument, self._source.wheel)

        await self._ssh.put(bootstrap.SCRIPT, bootstrap.script(self._image, self._source.argument).encode())
        await self._ssh.run(f"chmod +x {bootstrap.SCRIPT} && nohup {bootstrap.SCRIPT} > /dev/null 2>&1 &")

        await self._reach("bootstrap", BOOTSTRAP_TIMEOUT)

    async def _launch(self) -> None:
        """Start the worker, and open the way to it.

        The seeds arrive from above, and so do the rank and the peers. What a node
        is among the others is a fact about the compute — the one class of thing
        here a node cannot know about itself, and is not asked to.
        """
        environment = " ".join(
            f"{name}={shlex.quote(value)}"
            for name, value in (
                ("SKYWARD_NODE", self._machine.id),
                ("SKYWARD_COMPUTE", self._compute),
                ("SKYWARD_PEER", self.peer),
                ("SKYWARD_RANK", str(self._rank)),
                ("SKYWARD_PEERS", ",".join(self._peers)),
                ("SKYWARD_SEEDS", ",".join(self._seeds)),
                ("SKYWARD_SLOTS", str(self._concurrency)),
                ("SKYWARD_PLUGINS", msgspec.json.encode(self._plugins).decode()),
            )
        )
        await self._ssh.run(
            f"nohup env {environment} {bootstrap.PYTHON} -m skyward.runtime.worker "
            f">> {SKYWARD_DIR}/worker.log 2>&1 &",
        )

        await self._reach("worker", WORKER_TIMEOUT)
        self.tunnel = await self._ssh.forward(worker.PORT)

    async def _reach(self, phase: str, timeout: float) -> None:
        async with asyncio.timeout(timeout):
            if error := await self._waiter(phase):
                raise BootstrapFailedError(error)

    def _waiter(self, phase: str) -> asyncio.Future[str | None]:
        if phase not in self._reached:
            waiter: asyncio.Future[str | None] = asyncio.get_running_loop().create_future()
            if self._failure:
                waiter.set_result(self._failure)
            self._reached[phase] = waiter
        return self._reached[phase]

    async def _watch(self) -> None:
        """Follow the node's event log for as long as the node lives.

        It survives the bootstrap ending and the worker starting, because it
        follows the file and not the process. A dropped link ends the tail; the
        loop resumes from the line it had reached, and nothing said in between is
        lost — it is on the machine's disk, not in flight.
        """
        line = 1
        while True:
            async for seen, event in events(self._ssh, first=line):
                line = seen + 1
                self._observe(event)
            await asyncio.sleep(1.0)

    def _observe(self, event: NodeEvent) -> None:
        match event:
            case Console(content=content, task=task):
                self._output(content, task)
            case Phase(event="completed", phase=phase):
                self._settle(phase, None)
            case Phase(event="failed", phase=phase, error=error):
                self._abort(f"{phase}: {error or 'failed'}")
            case Phase():
                pass

    def _settle(self, phase: str, error: str | None) -> None:
        waiter = self._waiter(phase)
        if not waiter.done():
            waiter.set_result(error)

    def _abort(self, error: str) -> None:
        """One failed phase fails everything still being waited on.

        A bootstrap that died in `deps` will never emit `bootstrap completed`, and
        a caller waiting for it would learn nothing until its timeout — which is
        the difference between a node that reports a broken package and a node that
        hangs for fifteen minutes.
        """
        self._failure = error
        for phase in self._reached:
            self._settle(phase, error)
