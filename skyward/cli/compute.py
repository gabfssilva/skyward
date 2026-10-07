"""``sky compute`` — the computes endpoint, as a command.

Every command here is one or two HTTP calls and a table. What a compute *is*
lives in the daemon; the flags only say which one, and creating one is a
``ComputeSpec`` posted whole rather than a decision taken locally.

Credentials are the exception, and only because they have to be: the daemon
never reads the environment, so the provider row is registered from this
process — the same thing the SDK does before it creates a pool.

``sky run`` is the other exception. It holds the compute its script declares for
as long as the script runs, through the same ``Compute`` a script written against
the SDK would open.
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from collections.abc import Callable, Sequence
from contextlib import aclosing, suppress
from functools import partial
from pathlib import Path
from typing import Annotated, Literal

import msgspec
from cyclopts import Parameter

from skyward import api
from skyward.api.v1 import ComputeResource, FunctionResource, NodeResource, Page, TaskResource
from skyward.cli import app as cli
from skyward.cli import compute_app
from skyward.cli._client import Work, call, resolve
from skyward.cli._output import Output, dump, render
from skyward.cli.script import FACTORIES, Call, Script, Whole, read
from skyward.core import console, usercode, writes
from skyward.core.client import Client
from skyward.core.compute import Compute
from skyward.core.errors import SkywardError
from skyward.core.provider import Provider
from skyward.core.provider import resolve as resolve_provider
from skyward.core.spec import bounds
from skyward.core.view import ComputeView
from skyward.shared import codec
from skyward.shared.events import ConsoleEvent, Event
from skyward.shared.observability import notice
from skyward.shared.schemas import (
    ComputeCreate,
    ComputeSpec,
    ComputeSpecPatch,
    Dispatch,
    FunctionExcerpt,
    Image,
    NodeBounds,
    PipIndex,
    PluginRef,
    ProviderCreate,
    ProviderRef,
    Spec,
    TaskCreate,
)
from skyward.worker.plugins import PLUGINS
from skyward.worker.script import Exited, Outcome, Returned
from skyward.worker.script import call as invoke
from skyward.worker.script import run as execute

type Where = Literal["all", "any"] | int
"""Where a script runs: every node, one with a slot going spare, or the one at a rank."""

COMPUTE_COLUMNS = ("id", "name", "state", "ready", "total", "generation", "created")
NODE_COLUMNS = ("id", "rank", "state", "desired", "machine", "address", "accelerator", "$/h")
IMAGE_COLUMNS = ("image", "value")
RAN_COLUMNS = ("node", "exit", "error")
WRITTEN_COLUMNS = ("node", "error")

BYTES = "application/octet-stream"
NODE_HELP = "Which nodes to reach: all, or a rank"
WHERE_HELP = "Where to run it: all, any (one node with a slot free), or a rank"
WAIT = 30
"""Seconds the daemon holds a result request open before answering nothing yet."""
IDLE = 1.0
"""Seconds of quiet that mean a settled task has no more output coming."""
DRAIN = 5.0
"""Longest a settled task waits on its own output before giving up on the rest."""


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


@compute_app.command(name="list")
def list_computes(
    *,
    state: Annotated[str | None, Parameter(help="Only computes in this state")] = None,
    history: Annotated[int, Parameter(help="How many finished computes to show under the live ones. 0 shows none")] = 5,
    url: Annotated[str | None, Parameter(help="Daemon URL")] = None,
    output: Annotated[Output, Parameter(help="table or json")] = "table",
) -> None:
    """List the computes the daemon knows about, newest first.

    A daemon keeps a compute's row long after its machines are gone, so the
    finished ones are all it accumulates. They are not what a person is looking
    at: the live ones come first and whole, and the finished ones follow as the
    newest ``--history`` of them.

    ``--state`` asks for one state instead, and is then the whole filtered list.
    """
    async def work(client: Client) -> tuple[ComputeResource, ...]:
        if state:
            return (await client.call("GET", "/v1/computes", Page[ComputeResource], state=state)).items

        live = await client.call("GET", "/v1/computes", Page[ComputeResource], live=True)
        if history <= 0:
            return live.items

        finished = await client.call("GET", "/v1/computes", Page[ComputeResource], live=False, limit=history)
        return live.items + finished.items

    render(COMPUTE_COLUMNS, [_compute_row(compute) for compute in _call(work, url=url)], output=output)


@compute_app.command(name="get")
def get_compute(
    ref: str,
    *,
    url: Annotated[str | None, Parameter(help="Daemon URL")] = None,
    output: Annotated[Output, Parameter(help="table or json")] = "table",
) -> None:
    """Read one compute, by id or by name."""
    compute = _call(lambda client: client.call("GET", f"/v1/computes/{ref}", ComputeResource), url=url)
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
    base: Annotated[str | None, Parameter(help="Docker image the machines start from")] = None,
    python: Annotated[str | None, Parameter(help="Python version the nodes run (3.12, 3.13, …)")] = None,
    pip: Annotated[list[str] | None, Parameter(help="A pip package to install, as pip spells it. Repeat for more than one")] = None,
    apt: Annotated[list[str] | None, Parameter(help="An apt package to install. Repeat for more than one")] = None,
    pip_index: Annotated[list[str] | None, Parameter(help="An extra package index URL. Repeat for more than one")] = None,
    env: Annotated[list[str] | None, Parameter(help="An environment variable on the nodes, as KEY=VALUE. Repeat for more than one")] = None,
    mutable: Annotated[bool, Parameter(negative="", help="Let `sky compute update` change the image's packages and includes while the compute is up")] = False,
    plugin: Annotated[list[str] | None, Parameter(help="A plugin, as NAME or NAME:key=value,… (torch, torch:backend=gloo). Repeat for more than one")] = None,
    url: Annotated[str | None, Parameter(help="Daemon URL")] = None,
    output: Annotated[Output, Parameter(help="table or json")] = "table",
) -> None:
    """Create a compute and return without waiting for it to be ready.

    ``--base``, ``--python``, ``--pip``, ``--apt``, ``--pip-index`` and ``--env`` are
    the image the nodes build, with the meaning ``Image`` gives them in the SDK: the
    packages land in the interpreter ``run`` and ``exec`` use. ``--mutable`` is
    ``Image(mutable=True)``: the packages and includes may be changed afterwards
    with ``update``, on the machines already up. ``--plugin`` names a
    plugin by its kind, with parameters as ``key=value`` after a colon, and is
    validated here against the plugin's own fields, the way constructing one in
    the SDK would.

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
                provider=ProviderRef(kind=account.kind, name=account.name or account.kind),
                accelerator=resolve(accelerator, None)[0],
                cpus=cpus,
                memory_gb=memory,
                region=region,
            ),
        ),
        nodes=NodeBounds(initial=nodes),
        image=Image(
            base=base,
            python=python,
            pip=tuple(pip or ()),
            apt=tuple(apt or ()),
            pip_indexes=tuple(PipIndex(url=index) for index in pip_index or ()),
            env=pairs(env or (), "--env"),
            mutable=mutable,
        ),
        plugins=tuple(_plugin(text) for text in plugin or ()),
    )
    if ttl is not None:
        spec = msgspec.structs.replace(spec, ttl=ttl)

    async def work(client: Client) -> ComputeResource:
        await _register(client, account)
        return await client.call(
            "POST",
            "/v1/computes",
            ComputeResource,
            body=msgspec.json.encode(ComputeCreate(spec=spec, name=name)),
            headers={"Idempotency-Key": uuid.uuid4().hex},
        )

    render(COMPUTE_COLUMNS, [_compute_row(_call(work, url=url))], output=output)


@compute_app.command(name="scale")
def scale_compute(
    ref: str,
    *,
    nodes: Annotated[str, Parameter(help="How many machines: N, or MIN:MAX for an elastic range")],
    url: Annotated[str | None, Parameter(help="Daemon URL")] = None,
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
        lambda client: writes.conditional(client, ref, "PATCH", msgspec.json.encode(ComputeSpecPatch(nodes=wanted))),
        url=url,
    )
    render(COMPUTE_COLUMNS, [_compute_row(scaled)], output=output)


@compute_app.command(name="update")
def update_compute(
    ref: str,
    *,
    pip: Annotated[list[str] | None, Parameter(help="Packages the nodes install from here on; replaces the list")] = None,
    pip_index: Annotated[list[str] | None, Parameter(help="Extra package index URLs; replaces the list")] = None,
    include: Annotated[list[str] | None, Parameter(help="Paths to pack and ship, from the working directory; replaces the list")] = None,
    exclude: Annotated[list[str] | None, Parameter(help="Patterns left out of the includes; replaces the list")] = None,
    url: Annotated[str | None, Parameter(help="Daemon URL")] = None,
    output: Annotated[Output, Parameter(help="table or json")] = "table",
) -> None:
    """Change the packages or the code a mutable compute's nodes run.

    Each flag replaces the list the compute has, and a flag left out keeps it. The
    nodes that are up go through bootstrapping again and come back ready with the
    new image, and a task submitted meanwhile waits for them. So this returns a new
    ``generation`` rather than a finished update.

    A compute whose image was not created ``mutable`` refuses the change with
    ``image_fixed``: its image is its identity, and another one is another compute.
    """

    async def work(client: Client) -> ComputeResource:
        found = await client.call("GET", f"/v1/computes/{ref}", ComputeResource)
        image = msgspec.convert(msgspec.to_builtins(found.spec.image), Image)
        if pip is not None:
            image = msgspec.structs.replace(image, pip=tuple(pip))
        if pip_index is not None:
            image = msgspec.structs.replace(image, pip_indexes=tuple(PipIndex(url=index) for index in pip_index))
        if exclude is not None:
            image = msgspec.structs.replace(image, excludes=tuple(exclude))
        if include is not None or exclude is not None:
            includes = tuple(include) if include is not None else image.includes
            sha: str | None = None
            if includes:
                blob = await asyncio.to_thread(usercode.tarball, includes, image.excludes)
                sha = await codec.digest(blob)
                await client.upload(f"/v1/blobs/{sha}", blob)
            image = msgspec.structs.replace(image, includes=includes, includes_sha256=sha)
        return await writes.conditional(client, found.id, "PATCH", body=msgspec.json.encode(ComputeSpecPatch(image=image)))

    render(COMPUTE_COLUMNS, [_compute_row(_call(work, url=url))], output=output)


@compute_app.command(name="delete")
def delete_compute(
    ref: str,
    *,
    url: Annotated[str | None, Parameter(help="Daemon URL")] = None,
    output: Annotated[Output, Parameter(help="table or json")] = "table",
) -> None:
    """Mark a compute for destruction.

    The delete is accepted, not done: reconciliation runs until the provider
    confirms the machines are gone, so what comes back is still ``deleting``.
    """
    key = uuid.uuid4().hex
    deleted = _call(lambda client: writes.conditional(client, ref, "DELETE", headers={"Idempotency-Key": key}), url=url)
    render(COMPUTE_COLUMNS, [_compute_row(deleted)], output=output)


@compute_app.command(name="view")
def view_compute(
    ref: str,
    *,
    url: Annotated[str | None, Parameter(help="Daemon URL")] = None,
    output: Annotated[Output, Parameter(help="table or json")] = "table",
) -> None:
    """Read a compute together with the machines it is standing on."""

    compute = _call(lambda client: client.call("GET", f"/v1/computes/{ref}?include=nodes.replaced", ComputeResource), url=url)
    render(COMPUTE_COLUMNS, [_compute_row(compute)], output=output)
    if image := _image_rows(compute.spec):
        render(IMAGE_COLUMNS, image, output=output)
    render(NODE_COLUMNS, [_node_row(node) for node in compute.nodes], output=output)


@compute_app.command(name="ls")
def list_path(
    ref: str,
    path: str,
    *,
    node: Annotated[str, Parameter(name="--node", help=NODE_HELP)] = "0",
    url: Annotated[str | None, Parameter(help="Daemon URL")] = None,
    output: Annotated[Output, Parameter(help="table or json")] = "table",
) -> None:
    """List a path on the compute's nodes."""
    target = _node(node)
    ran = _call(
        lambda client: client.call("GET", f"/v1/computes/{ref}/files", dict[str, Result], path=path, node=target),
        url=url,
    )
    _spoke(ran, output)


@compute_app.command(name="rm")
def remove_path(
    ref: str,
    path: str,
    *,
    node: Annotated[str, Parameter(name="--node", help=NODE_HELP)] = "all",
    url: Annotated[str | None, Parameter(help="Daemon URL")] = None,
    output: Annotated[Output, Parameter(help="table or json")] = "table",
) -> None:
    """Remove a path on the compute's nodes, recursively."""
    target = _node(node)
    ran = _call(
        lambda client: client.call("DELETE", f"/v1/computes/{ref}/files", dict[str, Result], path=path, node=target),
        url=url,
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

    sys.stdout.write(f"{local}  {_call(work, url=url)} bytes\n")


@compute_app.command(name="exec")
def exec_command(
    ref: str,
    command: Annotated[list[str], Parameter(help="The command line, run by the node's shell")],
    *,
    node: Annotated[str, Parameter(name="--node", help=NODE_HELP)] = "all",
    url: Annotated[str | None, Parameter(help="Daemon URL")] = None,
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
    node: Annotated[str, Parameter(name="--node", help=WHERE_HELP)] = "any",
    url: Annotated[str | None, Parameter(help="Daemon URL")] = None,
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

    where = _where(node)
    source = script.read_text()
    work = partial(execute, source, (str(script), *(args or ())))
    if status := max(_call(lambda client: _remotely(client, ref, work, source, script.name, where), url=url), default=0):
        raise SystemExit(status)


def run_declared(
    script: Path | None = None,
    *args: Annotated[str, Parameter(allow_leading_hyphen=True, help="The function's command line, or the script's argv")],
    node: Annotated[str, Parameter(name="--node", help=WHERE_HELP)] = "any",
    url: Annotated[str | None, Parameter(help="Daemon URL")] = None,
) -> None:
    """Run a local Python file on the compute it declares itself.

    A file declares its compute in its PEP 723 header — ``requires-python`` and
    ``dependencies`` are the image, ``[tool.skyward]`` the rest — and then all of
    it runs, with ``args`` as its argv. Or it decorates functions with
    ``sky.app``, and then ``args`` name one and are its arguments, parsed against
    its signature; ``sky run FILE --help`` lists them. What the function returns
    is printed as JSON once it has, and as a list by rank under ``--node all``.

    The compute is named after the file and what it declares, so a second run
    lands on the same machines while they are up, and a declaration whose
    ``nodes`` changed resizes them rather than buying others. Whether they are
    still up after a run is its ``delete_on_exit``.

    What the image includes goes with each run, not with the machines: they are
    built without it, so a node never holds a copy older than the one it runs.

    Exits with the worst node's status. Given no file, prints this.
    """
    if script is None:
        cli.help_print(["run"])
        return
    if not script.is_file():
        raise SystemExit(f"no such script: {script}")

    where = _where(node)
    declared = read(script, args)
    app = declared.app
    daemon = resolve(url)
    standing = _call(lambda client: _standing(client, declared), url=daemon)
    pool = (
        Compute.attached(declared.name, url=daemon, console=False, callbacks=(_progress,), delete_on_exit=app.delete_on_exit)
        if standing
        else Compute(
            provider=app.provider,
            accelerator=app.accelerator,
            cpus=app.cpus,
            memory_gb=app.memory_gb,
            region=app.region,
            nodes=app.nodes,
            allocation=app.allocation,
            image=_without_includes(app.image),
            plugins=app.plugins,
            options=app.options,
            name=declared.name,
            url=daemon,
            delete_on_exit=app.delete_on_exit,
            console=False,
            callbacks=(_progress,),
        )
    )

    try:
        with pool:
            status = pool.loop.run(_performed(pool.client, pool.id, declared, where))
    except SkywardError as error:
        raise SystemExit(f"{error.code}: {error.message}") from None

    if not app.delete_on_exit:
        notice("WARNING", f"{declared.name} is still up")
    if status:
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
    ``max``, and an ``initial`` moved on its own would be a write that changes nothing.
    """
    match value.split(":"):
        case [count] if count.isdigit():
            return bounds(int(count))
        case [minimum, maximum] if minimum.isdigit() and maximum.isdigit() and int(minimum) <= int(maximum):
            return bounds((int(minimum), int(maximum)))
        case _:
            raise SystemExit(f"--nodes takes N, or MIN:MAX with MIN <= MAX, not {value!r}")


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


async def _standing(client: Client, script: Script) -> bool:
    """Whether the compute the file declares is up, sized and imaged to what it asks for now.

    A mutable image is brought to the one the file asks for, keeping whatever the
    compute was given to include by ``sky compute update`` — a run ships its own
    includes and has no say over those. A fixed image is part of the compute's name,
    so it is already the same. A deleted compute is not standing, and its name is
    free again. One still being deleted is refused rather than waited on: its
    machines are on their way out, and the name is not free until they are gone.
    """
    try:
        found = await client.call("GET", f"/v1/computes/{script.name}", ComputeResource)
    except SkywardError as error:
        if error.code != "not_found":
            raise
        return False

    match found.status.state:
        case "deleted":
            return False
        case "deleting":
            raise SystemExit(f"compute {script.name} is being deleted; run again once it is gone")
        case _:
            wanted = bounds(script.app.nodes)
            resized = (found.spec.nodes.initial, found.spec.nodes.min, found.spec.nodes.max) != (wanted.initial, wanted.min, wanted.max)
            current = msgspec.convert(msgspec.to_builtins(found.spec.image), Image)
            image = msgspec.structs.replace(
                _without_includes(script.app.image),
                includes=current.includes,
                excludes=current.excludes,
                includes_sha256=current.includes_sha256,
            )
            reimaged = current.mutable and current != image
            if resized or reimaged:
                patch = ComputeSpecPatch(nodes=wanted if resized else msgspec.UNSET, image=image if reimaged else msgspec.UNSET)
                await writes.conditional(client, found.id, "PATCH", msgspec.json.encode(patch))
            return True


def _without_includes(image: Image) -> Image:
    """The image a run creates: the compute's, with the code left to each run that ships it."""
    return msgspec.structs.replace(image, includes=(), excludes=())


def _where(node: str) -> Where:
    """``--node`` as a placement, refused here rather than by a task that waits for a rank forever."""
    match node:
        case "all" | "any":
            return node
        case rank if rank.isdigit():
            return int(rank)
        case _:
            raise SystemExit(f"--node takes all, any or a rank, not {node!r}")


async def _performed(client: Client, compute: str, script: Script, where: Where) -> int:
    """Run what the file declares, print what a function returned, and answer with the worst status."""
    match script.work:
        case Whole(argv):
            statuses = await _remotely(client, compute, partial(execute, script.source, argv, script.includes), script.source, script.path.name, where)
            return max(statuses, default=0)
        case Call(entry, arguments):
            work = partial(invoke, script.source, str(script.path), entry, arguments, script.includes)
            outcomes = await _remotely(client, compute, work, script.source, entry, where)
            _answered(outcomes, where)
            return max((_exit(outcome) for outcome in outcomes), default=0)


async def _remotely[R](client: Client, ref: str, work: Callable[[], R], excerpt: str, name: str, where: Where) -> list[R]:
    """Submit ``work``, print what it prints, and answer with what each node returned, by rank.

    The compute is read first for its id: a task takes a reference, and the event
    log is per id — following the wrong one would print nothing and look like a
    script that said nothing.

    A result arriving does not mean the output has: the value comes back over one
    request and the lines over another, and cancelling the follower the moment the
    task settles drops whatever the stream had not delivered yet. So the follower
    is told the task is done and left to drain until the log goes quiet.
    """
    compute = await client.call("GET", f"/v1/computes/{ref}", ComputeResource)
    task = await _submit(client, compute.id, work, excerpt, name, where)

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

    return await _results(client, task.id, codec.Pickle[R]())


async def _submit(client: Client, compute: str, work: Callable[[], object], excerpt: str, name: str, where: Where) -> TaskResource:
    """``work`` as a task: :mod:`skyward.worker.script` bound to the file's text.

    The text is also the function's excerpt, so the console shows the file that
    ran rather than the few lines that ran it.
    """
    blob = await codec.payload.encode(work)
    function = await codec.digest(blob)
    await client.upload(f"/v1/functions/{function}", blob, headers={"X-Skyward-Function-Name": name})
    await client.call("PUT", f"/v1/functions/{function}/excerpt", FunctionResource, body=msgspec.json.encode(FunctionExcerpt(text=excerpt)))

    dispatch: Dispatch
    match where:
        case "all":
            dispatch, rank = "all", None
        case "any":
            dispatch, rank = "one", None
        case int(rank):
            dispatch = "one"

    return await client.call(
        "POST",
        "/v1/tasks",
        TaskResource,
        body=msgspec.json.encode(
            TaskCreate(
                compute=compute,
                function=function,
                dispatch=dispatch,
                args_inline=await codec.payload.encode(((), {})),
                rank=rank,
            ),
        ),
        headers={"Idempotency-Key": uuid.uuid4().hex},
    )


def _answered(outcomes: Sequence[Outcome], where: Where) -> None:
    """What the function returned, on stdout: the one value, or every node's in a list under ``--node all``.

    Nothing is printed when a node did not return — its traceback is already on
    the terminal and the status says so — nor when nothing was returned at all.
    """
    values = [value for outcome in outcomes if (value := _value(outcome)) is not None]
    if len(values) < len(outcomes) or all(value == b"null" for value in values):
        return
    answer = b"[" + b",".join(values) + b"]" if where == "all" else values[0]
    sys.stdout.write(f"{answer.decode()}\n")
    sys.stdout.flush()


def _value(outcome: Outcome) -> bytes | None:
    match outcome:
        case Returned(value):
            return value
        case Exited():
            return None


def _exit(outcome: Outcome) -> int:
    match outcome:
        case Returned():
            return 0
        case Exited(status):
            return status


def _progress(event: Event, _: ComputeView) -> None:
    """What the machines are doing, on stderr, while ``sky run`` holds them.

    A line a script printed is left out: those are the task's, and they go to
    stdout from the task itself, drained after it settles — the pool's own stream
    is cancelled the moment the block ends, and would drop the last of them.
    """
    match event:
        case ConsoleEvent(task=str()):
            return
        case _:
            if line := console.render(event, sys.stderr.isatty()):
                print(line, file=sys.stderr, flush=True)


async def _console(client: Client, compute: str, task: str, settled: asyncio.Event) -> None:
    """The lines this task's nodes wrote, as they write them.

    The log replays from its start, so subscribing after the task was submitted
    loses nothing — the first thing the script printed is still in it.

    A line carries the task it belongs to — the daemon resolves it from the attempt
    the machine was handed — so a compute running more than one is filtered on the
    line itself rather than by reading the task back per attempt.

    Every line is printed after the rank of the node that wrote it. A script's
    output and this command's own sentences share a terminal, and the lines of a
    broadcast arrive interleaved; without it neither can be told apart. The ranks
    are read off the compute, and read again for a node they do not know yet — one
    a resize brought in after the first read.
    """
    ranks: dict[str, int] = {}
    async with aclosing(client.events(compute, types=("node.console",))) as stream:
        feed = stream.__aiter__()
        while True:
            try:
                async with asyncio.timeout(IDLE if settled.is_set() else None):
                    _, payload = await anext(feed)
            except (TimeoutError, StopAsyncIteration):
                return

            line = json.loads(payload)
            if line.get("task") != task:
                continue
            node = line.get("node", "")
            if node not in ranks:
                current = await client.call("GET", f"/v1/computes/{compute}", ComputeResource)
                ranks = {known.id: known.rank for known in current.nodes}
            sys.stdout.write(f"{ranks.get(node, node)} │ {line.get('content', '')}\n")
            sys.stdout.flush()


async def _results[R](client: Client, task_id: str, reading: codec.Pickle[R]) -> list[R]:
    """What each node's execution returned, in rank order. An execution that returned nothing — lost, or stopped — has no say."""
    settled = await client.call("GET", f"/v1/tasks/{task_id}", TaskResource)
    return [
        await reading.decode(blob)
        for execution in sorted(settled.executions, key=lambda execution: execution.rank)
        if execution.result_sha256 and (blob := await client.blob(f"/v1/blobs/{execution.result_sha256}"))
    ]


def pairs(values: Sequence[str], flag: str) -> dict[str, str]:
    """``key=value`` as a mapping, refusing anything that is not one."""
    written: dict[str, str] = {}
    for pair in values:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise SystemExit(f"{flag} takes key=value, not {pair!r}")
        written[key] = value
    return written


def _plugin(text: str) -> PluginRef:
    """``name`` or ``name:key=value,…`` as a reference, checked against the plugin it names."""
    kind, _, given = text.partition(":")
    if kind not in PLUGINS:
        raise SystemExit(f"unknown plugin '{kind}'; known: {', '.join(sorted(PLUGINS))}")

    try:
        return msgspec.convert(pairs(given.split(",") if given else (), "--plugin"), PLUGINS[kind], strict=False).ref()
    except msgspec.ValidationError as invalid:
        raise SystemExit(f"--plugin {kind}: {invalid}") from None


def _image_rows(spec: api.v1.ComputeSpec) -> list[tuple[str, str]]:
    """The image a compute asked for, one row per field that says something."""
    image = spec.image
    listed = (
        ("base", image.base),
        ("python", image.python),
        ("pip", " ".join(image.pip)),
        ("apt", " ".join(image.apt)),
        ("pip_indexes", " ".join(index.url for index in image.pip_indexes)),
        ("env", " ".join(f"{key}={value}" for key, value in image.env.items())),
        ("plugins", " ".join(_plugin_text(ref) for ref in spec.plugins)),
    )
    return [(field, value) for field, value in listed if value]


def _plugin_text(ref: api.v1.PluginRef) -> str:
    given = ",".join(f"{key}={value}" for key, value in ref.params.items() if value is not None)
    return f"{ref.kind}:{given}" if given else ref.kind


def _compute_row(compute: ComputeResource) -> tuple[object, ...]:
    ready = sum(node.state == "ready" for node in compute.nodes)
    return (compute.id, compute.name, compute.status.state, ready, len(compute.nodes), compute.generation, compute.created_at)


def _node_row(node: NodeResource) -> tuple[object, ...]:
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


def _call[T](work: Work[T], *, url: str | None = None) -> T:
    """Run the work, and turn a refusal into a message rather than a traceback."""
    try:
        return call(work, url=url)
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
    "pairs",
    "remove_path",
    "run_declared",
    "run_script",
    "upload_path",
    "view_compute",
]
