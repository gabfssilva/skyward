"""``sky compute`` — the computes endpoint, as a command.

Every command here is one or two HTTP calls and a table. What a compute *is*
lives in the daemon; the flags only say which one, and creating one is a
``ComputeSpec`` posted whole rather than a decision taken locally.

Credentials are the exception, and only because they have to be: the daemon
never reads the environment, so the provider row is registered from this
process — the same thing the SDK does before it creates a pool.
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from collections.abc import Callable
from contextlib import aclosing, suppress
from pathlib import Path
from typing import Annotated

import msgspec
from cyclopts import Parameter

from skyward.cli import compute_app
from skyward.cli._client import Work, call
from skyward.cli._output import Output, dump, render
from skyward.core import provider as factories
from skyward.core.client import Client
from skyward.core.errors import SkywardError
from skyward.core.provider import Provider
from skyward.core.provider import resolve as resolve_provider
from skyward.core.spec import bounds
from skyward.shared import codec
from skyward.shared.schemas import (
    Compute,
    ComputeCreate,
    ComputeSpec,
    ComputeSpecPatch,
    Dispatch,
    Node,
    NodeBounds,
    Page,
    ProviderCreate,
    ProviderRef,
    Spec,
    Task,
    TaskCreate,
)

COMPUTE_COLUMNS = ("id", "name", "state", "ready", "total", "generation", "created")
NODE_COLUMNS = ("id", "rank", "state", "desired", "machine", "address", "accelerator", "$/h")
RAN_COLUMNS = ("node", "exit", "error")
WRITTEN_COLUMNS = ("node", "error")

BYTES = "application/octet-stream"
NODE_HELP = "Which nodes to reach: all, or a rank"
WAIT = 30
"""Seconds the daemon holds a result request open before answering nothing yet."""
IDLE = 1.0
"""Seconds of quiet that mean a settled task has no more output coming."""
DRAIN = 5.0
"""Longest a settled task waits on its own output before giving up on the rest."""
WRITE_ATTEMPTS = 5
"""How many times a conditional write is re-read and re-sent before it is a real conflict."""


class Result(msgspec.Struct, frozen=True):
    """What one node said, as the daemon reports it.

    Restated here rather than imported: the daemon's own copy lives next to its
    SSH channel, which imports ``asyncssh``, and the CLI is installable without
    the server extra — a command talking to a remote daemon has no business
    needing the library that daemon dials machines with.
    """

    exit_code: int
    stdout: str
    stderr: str


FACTORIES: dict[str, Callable[[], Provider]] = {
    "aws": factories.AWS,
    "container": factories.Container,
    "gcp": factories.GCP,
    "hyperstack": factories.Hyperstack,
    "jarvislabs": factories.JarvisLabs,
    "lambda": factories.Lambda,
    "massed_compute": factories.MassedCompute,
    "novita": factories.Novita,
    "runpod": factories.RunPod,
    "scaleway": factories.Scaleway,
    "tensordock": factories.TensorDock,
    "vastai": factories.VastAI,
    "verda": factories.Verda,
    "vultr": factories.Vultr,
}


@compute_app.command(name="list")
def list_computes(
    *,
    state: Annotated[str | None, Parameter(help="Only computes in this state")] = None,
    url: Annotated[str | None, Parameter(help="Daemon URL")] = None,
    database: Annotated[Path | None, Parameter(help="Embedded daemon database")] = None,
    output: Annotated[Output, Parameter(help="table or json")] = "table",
) -> None:
    """List the computes the daemon knows about."""
    page = _call(lambda client: client.call("GET", "/v1/computes", Page[Compute], state=state), url=url, database=database)
    render(COMPUTE_COLUMNS, [_compute_row(compute) for compute in page.items], output=output)


@compute_app.command(name="get")
def get_compute(
    ref: str,
    *,
    url: Annotated[str | None, Parameter(help="Daemon URL")] = None,
    database: Annotated[Path | None, Parameter(help="Embedded daemon database")] = None,
    output: Annotated[Output, Parameter(help="table or json")] = "table",
) -> None:
    """Read one compute, by id or by name."""
    compute = _call(lambda client: client.call("GET", f"/v1/computes/{ref}", Compute), url=url, database=database)
    render(COMPUTE_COLUMNS, [_compute_row(compute)], output=output)


@compute_app.command(name="create")
def create_compute(
    *,
    provider: Annotated[str, Parameter(help="Provider kind (aws, runpod, vastai, …)")],
    name: Annotated[str | None, Parameter(help="Name to reach this compute by")] = None,
    accelerator: Annotated[str | None, Parameter(help="Accelerator to ask for (A100, H100, …)")] = None,
    nodes: Annotated[int, Parameter(help="How many machines")] = 1,
    region: Annotated[str | None, Parameter(help="Where to buy them")] = None,
    cpus: Annotated[int | None, Parameter(help="Least vCPUs per machine")] = None,
    memory: Annotated[int | None, Parameter(help="Least memory per machine, in GB")] = None,
    ttl: Annotated[int | None, Parameter(help="Seconds a machine may sit with nobody connected before it removes itself. 0 never does")] = None,
    url: Annotated[str | None, Parameter(help="Daemon URL")] = None,
    database: Annotated[Path | None, Parameter(help="Embedded daemon database")] = None,
    output: Annotated[Output, Parameter(help="table or json")] = "table",
) -> None:
    """Create a compute and return without waiting for it to be ready.

    ``--ttl`` is the dead-man switch the providers that support one arm on each
    machine: with nobody connected for that long, the machine takes itself away
    rather than billing for a daemon that is never coming back. The default is the
    spec's, and it is short — a compute whose machines are slow to become
    reachable is a compute whose machines can reach it first.
    """
    from skyward.shared.accelerators import resolve

    if provider not in FACTORIES:
        raise SystemExit(f"unknown provider '{provider}'; known: {', '.join(sorted(FACTORIES))}")

    account = FACTORIES[provider]()
    spec = ComputeSpec(
        specs=(
            Spec(
                provider=ProviderRef(kind=account.kind),
                accelerator=resolve(accelerator, None)[0],
                cpus=cpus,
                memory_gb=memory,
                region=region,
            ),
        ),
        nodes=NodeBounds(desired=nodes),
    )
    if ttl is not None:
        spec = msgspec.structs.replace(spec, ttl=ttl)

    async def work(client: Client) -> Compute:
        await _register(client, account)
        return await client.call(
            "POST",
            "/v1/computes",
            Compute,
            body=msgspec.json.encode(ComputeCreate(spec=spec, name=name)),
            headers={"Idempotency-Key": uuid.uuid4().hex},
        )

    render(COMPUTE_COLUMNS, [_compute_row(_call(work, url=url, database=database))], output=output)


@compute_app.command(name="scale")
def scale_compute(
    ref: str,
    *,
    nodes: Annotated[str, Parameter(help="How many machines: N, or MIN:MAX for an elastic range")],
    url: Annotated[str | None, Parameter(help="Daemon URL")] = None,
    database: Annotated[Path | None, Parameter(help="Embedded daemon database")] = None,
    output: Annotated[Output, Parameter(help="table or json")] = "table",
) -> None:
    """Change how many machines a compute stands on.

    A size is the one part of a definition that changes without replacing anything:
    what is up is kept, and the difference is bought or drained by reconciliation.
    So this returns a new ``generation`` rather than a finished resize — the machines
    arrive, or leave, afterwards.
    """
    wanted = _nodes(nodes)
    scaled = _call(
        lambda client: _conditional(client, ref, "PATCH", msgspec.json.encode(ComputeSpecPatch(nodes=wanted))),
        url=url,
        database=database,
    )
    render(COMPUTE_COLUMNS, [_compute_row(scaled)], output=output)


@compute_app.command(name="delete")
def delete_compute(
    ref: str,
    *,
    url: Annotated[str | None, Parameter(help="Daemon URL")] = None,
    database: Annotated[Path | None, Parameter(help="Embedded daemon database")] = None,
    output: Annotated[Output, Parameter(help="table or json")] = "table",
) -> None:
    """Mark a compute for destruction.

    The delete is accepted, not done: reconciliation runs until the provider
    confirms the machines are gone, so what comes back is still ``deleting``.
    """
    key = uuid.uuid4().hex
    deleted = _call(lambda client: _conditional(client, ref, "DELETE", headers={"Idempotency-Key": key}), url=url, database=database)
    render(COMPUTE_COLUMNS, [_compute_row(deleted)], output=output)


@compute_app.command(name="view")
def view_compute(
    ref: str,
    *,
    url: Annotated[str | None, Parameter(help="Daemon URL")] = None,
    database: Annotated[Path | None, Parameter(help="Embedded daemon database")] = None,
    output: Annotated[Output, Parameter(help="table or json")] = "table",
) -> None:
    """Read a compute together with the machines it is standing on."""

    async def work(client: Client) -> tuple[Compute, Page[Node]]:
        compute = await client.call("GET", f"/v1/computes/{ref}", Compute)
        return compute, await client.call("GET", f"/v1/computes/{compute.id}/nodes", Page[Node])

    compute, nodes = _call(work, url=url, database=database)
    render(COMPUTE_COLUMNS, [_compute_row(compute)], output=output)
    render(NODE_COLUMNS, [_node_row(node) for node in nodes.items], output=output)


@compute_app.command(name="ls")
def list_path(
    ref: str,
    path: str,
    *,
    node: Annotated[str, Parameter(name="--node", help=NODE_HELP)] = "0",
    url: Annotated[str | None, Parameter(help="Daemon URL")] = None,
    database: Annotated[Path | None, Parameter(help="Embedded daemon database")] = None,
    output: Annotated[Output, Parameter(help="table or json")] = "table",
) -> None:
    """List a path on the compute's nodes."""
    target = _node(node)
    ran = _call(
        lambda client: client.call("GET", f"/v1/computes/{ref}/files", dict[str, Result], path=path, node=target),
        url=url,
        database=database,
    )
    _spoke(ran, output)


@compute_app.command(name="rm")
def remove_path(
    ref: str,
    path: str,
    *,
    node: Annotated[str, Parameter(name="--node", help=NODE_HELP)] = "all",
    url: Annotated[str | None, Parameter(help="Daemon URL")] = None,
    database: Annotated[Path | None, Parameter(help="Embedded daemon database")] = None,
    output: Annotated[Output, Parameter(help="table or json")] = "table",
) -> None:
    """Remove a path on the compute's nodes, recursively."""
    target = _node(node)
    ran = _call(
        lambda client: client.call("DELETE", f"/v1/computes/{ref}/files", dict[str, Result], path=path, node=target),
        url=url,
        database=database,
    )
    render(RAN_COLUMNS, [(name, result.exit_code, result.stderr.strip() or None) for name, result in ran.items()], output=output)


@compute_app.command(name="upload")
def upload_path(
    ref: str,
    local: Path,
    remote: str,
    *,
    node: Annotated[str, Parameter(name="--node", help=NODE_HELP)] = "all",
    url: Annotated[str | None, Parameter(help="Daemon URL")] = None,
    database: Annotated[Path | None, Parameter(help="Embedded daemon database")] = None,
    output: Annotated[Output, Parameter(help="table or json")] = "table",
) -> None:
    """Write a local file onto the compute's nodes.

    Every node by default. A file a task will read has to be wherever the task
    lands, and which node that is belongs to the dispatcher.
    """
    target = _node(node)
    if not local.is_file():
        raise SystemExit(f"no such file: {local}")

    content = local.read_bytes()
    written = _call(
        lambda client: client.call(
            "PUT",
            f"/v1/computes/{ref}/files",
            dict[str, str | None],
            body=content,
            headers={"Content-Type": BYTES},
            path=remote,
            node=target,
        ),
        url=url,
        database=database,
    )
    render(WRITTEN_COLUMNS, list(written.items()), output=output)


@compute_app.command(name="download")
def download_path(
    ref: str,
    remote: str,
    local: Path,
    *,
    node: Annotated[str, Parameter(name="--node", help="Which node to read from: a rank")] = "0",
    url: Annotated[str | None, Parameter(help="Daemon URL")] = None,
    database: Annotated[Path | None, Parameter(help="Embedded daemon database")] = None,
) -> None:
    """Read a file off one of the compute's nodes.

    One node, named by rank. Four machines hold four files, and there is no
    answer to which of them the caller meant.
    """
    rank = _rank(node)

    async def work(client: Client) -> int:
        size = 0
        with local.open("wb") as sink:
            async for chunk in client.download(f"/v1/computes/{ref}/files/content", path=remote, node=rank):
                sink.write(chunk)
                size += len(chunk)
        return size

    sys.stdout.write(f"{local}  {_call(work, url=url, database=database)} bytes\n")


@compute_app.command(name="exec")
def exec_command(
    ref: str,
    command: Annotated[list[str], Parameter(help="The command line, run by the node's shell")],
    *,
    node: Annotated[str, Parameter(name="--node", help=NODE_HELP)] = "all",
    url: Annotated[str | None, Parameter(help="Daemon URL")] = None,
    database: Annotated[Path | None, Parameter(help="Embedded daemon database")] = None,
    output: Annotated[Output, Parameter(help="table or json")] = "table",
) -> None:
    """Run a shell command on the compute's nodes.

    The machine's shell, not the worker's: this answers questions about the node
    — what the driver reports, what is on the disk — and reaches one whose worker
    is busy. Running the user's code is what ``run`` is for.

    Exits with the worst node's status.
    """
    target = _node(node)
    ran = _call(
        lambda client: client.call("POST", f"/v1/computes/{ref}/exec", dict[str, Result], command=" ".join(command), node=target),
        url=url,
        database=database,
    )
    _spoke(ran, output)
    if worst := max((result.exit_code for result in ran.values()), default=0):
        raise SystemExit(worst)


@compute_app.command(name="run")
def run_script(
    ref: str,
    script: Path,
    args: Annotated[list[str] | None, Parameter(help="Forwarded to the script as sys.argv")] = None,
    *,
    every: Annotated[bool, Parameter(name="--all", help="Run it on every node rather than one")] = False,
    url: Annotated[str | None, Parameter(help="Daemon URL")] = None,
    database: Annotated[Path | None, Parameter(help="Embedded daemon database")] = None,
) -> None:
    """Run a local Python script on the compute.

    A task, not a shell command: the script is sent down the same path a
    ``@sky.function`` takes, so it lands in a worker with the image, the plugins
    and the runtime API around it, and what it prints comes back over the
    compute's event log as it prints it.

    Exits with the worst node's status.
    """
    if not script.is_file():
        raise SystemExit(f"no such script: {script}")

    source = script.read_text()
    argv = (str(script), *(args or ()))
    if status := _call(lambda client: _remotely(client, ref, source, argv, "all" if every else "one"), url=url, database=database):
        raise SystemExit(status)


def _node(node: str) -> str:
    """Reject a target the daemon would only reject later, and with less to say."""
    if node == "all" or node.lstrip("-").isdigit():
        return node
    raise SystemExit(f"--node takes 'all' or a rank, not {node!r}")


def _nodes(value: str) -> NodeBounds:
    """``N`` for a fixed size, ``MIN:MAX`` for an elastic range — ``nodes=`` as a flag.

    The bounds are written whole rather than field by field, because a size is only
    coherent as a set: on an elastic compute the reconciler sizes between ``min`` and
    ``max``, and a ``desired`` moved on its own would be a write that changes nothing.
    """
    match value.split(":"):
        case [count] if count.isdigit() and int(count) > 0:
            return bounds(int(count))
        case [minimum, maximum] if minimum.isdigit() and maximum.isdigit() and 0 < int(minimum) <= int(maximum):
            return bounds((int(minimum), int(maximum)))
        case _:
            raise SystemExit(f"--nodes takes N, or MIN:MAX with 1 <= MIN <= MAX, not {value!r}")


async def _conditional(client: Client, ref: str, method: str, body: bytes | None = None, headers: dict[str, str] | None = None) -> Compute:
    """A write against a compute, guarded by the revision it was read at.

    ``If-Match`` is what keeps two terminals from overwriting each other's intent,
    but the revision also moves on bookkeeping nobody asked for: the reconciler
    writes what it observed on every tick, and a lease renews itself on a timer.
    Either landing between the read and the write refuses a change nothing was
    racing, so the precondition is refreshed rather than handed back as a failure.
    """
    attempts = WRITE_ATTEMPTS
    while True:
        current = await client.call("GET", f"/v1/computes/{ref}", Compute)
        try:
            return await client.call(
                method,
                f"/v1/computes/{current.id}",
                Compute,
                body=body,
                headers={"If-Match": f'"{current.revision}"', **(headers or {})},
            )
        except SkywardError as error:
            attempts -= 1
            if error.code != "revision_conflict" or not attempts:
                raise


def _rank(node: str) -> str:
    if node.lstrip("-").isdigit():
        return node
    raise SystemExit(f"--node takes a rank, not {node!r}")


def _spoke(ran: dict[str, Result], output: Output) -> None:
    """Per node, what the command printed. A listing does not fit in a cell."""
    match output:
        case "json":
            dump([{"node": name, "exit": r.exit_code, "stdout": r.stdout, "stderr": r.stderr} for name, r in ran.items()])
        case "table":
            for name, result in ran.items():
                sys.stdout.write(f"{name}\n{(result.stdout + result.stderr).rstrip()}\n")


async def _remotely(client: Client, ref: str, source: str, argv: tuple[str, ...], dispatch: Dispatch) -> int:
    """Submit the script, print what it prints, and answer with its status.

    The compute is read first for its id: a task takes a reference, and the event
    log is per id — following the wrong one would print nothing and look like a
    script that said nothing.

    A result arriving does not mean the output has: the value comes back over one
    request and the lines over another, and cancelling the follower the moment the
    task settles drops whatever the stream had not delivered yet. So the follower
    is told the task is done and left to drain until the log goes quiet.
    """
    compute = await client.call("GET", f"/v1/computes/{ref}", Compute)
    task = await _submit(client, compute.id, source, argv, dispatch)

    settled = asyncio.Event()
    printing = asyncio.get_running_loop().create_task(_console(client, compute.id, task.id, settled))
    try:
        while await client.blob(f"/v1/tasks/{task.id}/result", wait=WAIT) is None:
            continue
    finally:
        settled.set()
        with suppress(TimeoutError):
            async with asyncio.timeout(DRAIN):
                await printing
        printing.cancel()
        with suppress(asyncio.CancelledError):
            await printing

    return await _status(client, task.id)


async def _submit(client: Client, compute: str, source: str, argv: tuple[str, ...], dispatch: Dispatch) -> Task:
    blob = await codec.payload.encode(_wrap(source, argv))
    function = await codec.digest(blob)
    await client.upload(f"/v1/functions/{function}", blob, headers={"X-Skyward-Function-Name": Path(argv[0]).name})

    return await client.call(
        "POST",
        "/v1/tasks",
        Task,
        body=msgspec.json.encode(
            TaskCreate(
                compute=compute,
                function=function,
                dispatch=dispatch,
                args_inline=await codec.payload.encode(((), {})),
            ),
        ),
        headers={"Idempotency-Key": uuid.uuid4().hex},
    )


async def _console(client: Client, compute: str, task: str, settled: asyncio.Event) -> None:
    """The lines this task's nodes wrote, as they write them.

    The log replays from its start, so subscribing after the task was submitted
    loses nothing — the first thing the script printed is still in it.

    A line names the *execution* that produced it, not the task: a task is one
    call, and a broadcast of it is one execution per node. So the task is read
    back for the executions it has, and only when a line names one that has not
    been decided yet — which is once per node, not once per line.
    """
    mine: set[str] = set()
    foreign: set[str] = set()

    async def belongs(execution: str) -> bool:
        nonlocal mine
        if execution in mine:
            return True
        if execution in foreign:
            return False

        mine = {run.id for run in (await client.call("GET", f"/v1/tasks/{task}", Task)).executions}
        if execution in mine:
            return True

        foreign.add(execution)
        return False

    async with aclosing(client.events(compute)) as stream:
        feed = stream.__aiter__()
        while True:
            try:
                async with asyncio.timeout(IDLE if settled.is_set() else None):
                    name, payload = await anext(feed)
            except (TimeoutError, StopAsyncIteration):
                return

            if name != "node.console":
                continue
            line = json.loads(payload)
            if (execution := line.get("task")) and await belongs(execution):
                sys.stdout.write(line.get("content", "") + "\n")
                sys.stdout.flush()


async def _status(client: Client, task_id: str) -> int:
    """The worst node's, because a broadcast is as good as its unhappiest machine."""
    settled = await client.call("GET", f"/v1/tasks/{task_id}", Task)
    codes = [
        await codec.Pickle[int]().decode(blob)
        for execution in settled.executions
        if execution.result_sha256 and (blob := await client.blob(f"/v1/blobs/{execution.result_sha256}"))
    ]
    return max(codes, default=0)


def _wrap(source: str, argv: tuple[str, ...]) -> Callable[[], int]:
    """The script as a function a worker can be handed, and its status back.

    A closure rather than a module-level function on purpose. Cloudpickle sends a
    module-level one by reference, and this module imports ``cyclopts``, which a
    node has no reason to have — the node would fail to unpickle it. Sent by
    value it carries its own code, and needs nothing there but the interpreter.
    """

    def execute() -> int:
        import sys
        import traceback

        held, sys.argv = sys.argv, list(argv)
        try:
            exec(compile(source, argv[0], "exec"), {"__name__": "__main__", "__file__": argv[0]})
        except SystemExit as stop:
            return stop.code if isinstance(stop.code, int) else (0 if stop.code is None else 1)
        except BaseException:
            traceback.print_exc()
            return 1
        finally:
            sys.argv = held
        return 0

    return execute


def _compute_row(compute: Compute) -> tuple[object, ...]:
    return (
        compute.id,
        compute.name,
        compute.status.state,
        compute.status.nodes_ready,
        compute.status.nodes_total,
        compute.generation,
        compute.created_at,
    )


def _node_row(node: Node) -> tuple[object, ...]:
    return (node.id, node.rank, node.state, node.desired, node.machine, node.address, node.accelerator, node.price_per_hour)


async def _register(client: Client, account: Provider) -> None:
    """Make sure the daemon has an account of this kind to log in with."""
    name = account.name or account.kind
    try:
        await client.call("GET", f"/v1/providers/{name}", dict[str, object])
    except SkywardError as error:
        if error.code != "not_found":
            raise
        credentials, config = resolve_provider(account)
        await client.call(
            "POST",
            "/v1/providers",
            dict[str, object],
            body=msgspec.json.encode(ProviderCreate(name=name, kind=account.kind, credentials=credentials, config=config)),
        )


def _call[T](work: Work[T], *, url: str | None = None, database: Path | None = None) -> T:
    """Run the work, and turn a refusal into a message rather than a traceback."""
    try:
        return call(work, url=url, database=database)
    except SkywardError as error:
        raise SystemExit(f"{error.code}: {error.message}") from None


__all__ = [
    "create_compute",
    "delete_compute",
    "download_path",
    "exec_command",
    "get_compute",
    "list_computes",
    "list_path",
    "remove_path",
    "run_script",
    "upload_path",
    "view_compute",
]
