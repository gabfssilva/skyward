"""``sky mcp`` — the daemon, as tools an AI assistant calls.

A Model Context Protocol server on stdin and stdout, started by the assistant's host
and kept for as long as its session lasts. It is the CLI in another grammar: each
tool makes the requests a ``sky`` command makes, against the daemon the CLI would
reach, and answers in JSON instead of a table. Nothing is decided here and nothing
is kept between two calls.

An answer is cut to what an assistant acts on, because everything else it is sent
is context it pays for: a compute, a task, an offer or a log line comes as the
fields that describe it, and nothing null or empty is sent. ``raw`` on a read is
the daemon's resource whole. Neither shows the values of a compute's environment or
its plugins' credentials — they are secrets as often as not, an assistant needs
their names rather than their values, and what it reads may be kept anywhere.

The annotations are what a host asks permission by: the reads say they only read,
and every tool that buys, changes or ends machines says so.
"""

from __future__ import annotations

import io
import time
from collections.abc import Awaitable, Callable, Collection, Mapping
from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime
from itertools import groupby
from pathlib import Path
from typing import Annotated, Literal

import httpx
import msgspec
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from msgspec import Struct
from msgspec.structs import asdict, replace
from pydantic import AwareDatetime, Field, JsonValue

from skyward.api import v1
from skyward.cli.compute import Result, compute_spec, computes, create, launch, update
from skyward.cli.script import FACTORIES, Script, read
from skyward.core import writes
from skyward.core.client import Client
from skyward.core.errors import SkywardError, UnexpectedResponseError
from skyward.core.spec import bounds
from skyward.shared import codec
from skyward.shared.providers import secrets
from skyward.shared.schemas import ComputeSpecPatch, Image, PipIndex, PluginRef
from skyward.shared.version import current
from skyward.worker.plugins import PLUGINS
from skyward.worker.script import Exited, Returned

INSTRUCTIONS = """\
Skyward rents machines from cloud providers and runs Python on them. A compute is a group of those machines (its \
nodes), bought, kept and deleted by the Skyward daemon these tools talk to; a task is one function call running on a \
compute; the event log holds what the nodes printed and every change of state. A compute is named by its id or by its \
name.

These tools are the `sky` command's, as calls. For what they cover, use them rather than `sky` in a shell, also in a \
session that has been running `sky` until now: they reach the same daemon and answer in less text. `sky compute exec` \
is exec_command; `sky compute list`, `sky status`, `get` and `view` are list_computes and get_compute; `sky compute \
tasks`, `cancel` and `result` are list_tasks, cancel_task and read_result; `sky compute create`, `scale`, `update` and `delete` (`sky stop`) \
are the tools of those names; `sky offers list` is list_offers; `sky providers list` is list_providers; `sky run` is \
run_script; the event log `sky log export` writes is read_log. The shell keeps what no tool does: files on the nodes, \
ssh, repl, `sky compute run`, `sky providers set`, `sky server start`, and following output as it arrives \
(`sky monitor`), which a tool answers only once.

create_compute, run_script, and scale_compute when it grows a compute, buy machines that cost money until \
delete_compute ends them. scale_compute when it shrinks a compute, update_compute and delete_compute act on machines \
that are running.

run_script runs a Python file on the compute it declares, as `sky run` does, and answers with the compute and the \
run's task as soon as the task is queued: read_log with the task gives what it prints, and read_result its exit status \
or what its function returned once it settles. The daemon holds the compute for the task, and deletes it once the task \
settles unless the file says `delete_on_exit = false`. `sky monitor <name>` in a background shell prints a line per \
event of the run as it happens, and ends once the compute is deleted.

An answer leaves out every field that is null or empty, but for a page's `items`. A compute, a task, an offer or a log line comes with the \
fields that matter to act on it; `raw` on the tool that read it answers with everything the daemon holds. The values \
of a compute's environment variables and its plugins' credentials are hidden either way.

Python that dispatches work to a compute is written against the Skyward SDK (`@sky.function`, `sky.Compute`), not \
through these tools."""

HIDDEN = "<hidden>"
"""What an answer carries in place of an environment variable's value or a plugin's credential."""

type Count = Annotated[int, Field(ge=0)]
type Positive = Annotated[int, Field(ge=1)]

READ = ToolAnnotations(read_only_hint=True, open_world_hint=False)
BUY = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=True)
CHANGE = ToolAnnotations(read_only_hint=False, destructive_hint=True, idempotent_hint=True, open_world_hint=True)
RUN = ToolAnnotations(read_only_hint=False, destructive_hint=True, idempotent_hint=False, open_world_hint=True)


def server(url: str) -> MCPServer:
    """The MCP server over the daemon at ``url``; ``run("stdio")`` serves it."""
    tools = Tools(url)
    served = MCPServer("skyward", version=current(), instructions=INSTRUCTIONS, log_level="WARNING")
    for tool, hints in (
        (tools.list_computes, READ),
        (tools.get_compute, READ),
        (tools.list_tasks, READ),
        (tools.read_log, READ),
        (tools.read_metrics, READ),
        (tools.read_metric_history, READ),
        (tools.read_daemon_log, READ),
        (tools.list_offers, READ),
        (tools.list_providers, READ),
        (tools.read_result, READ),
        (tools.create_compute, BUY),
        (tools.run_script, BUY),
        (tools.scale_compute, CHANGE),
        (tools.update_compute, CHANGE),
        (tools.delete_compute, CHANGE),
        (tools.cancel_task, CHANGE),
        (tools.exec_command, RUN),
    ):
        served.add_tool(tool, annotations=hints, structured_output=False)
    return served


class Tools:
    """Every tool, against the daemon at one address. A docstring here is what the assistant reads about its tool."""

    def __init__(self, url: str) -> None:
        self._url = url

    async def list_computes(self, state: v1.ComputeState | None = None, history: int = 5, raw: bool = False) -> str:
        """List the computes the daemon knows: every live one, then the newest finished ones.

        Each comes with its state, what it asked for and what was bought, how many nodes are ready, and what it costs;
        ``get_compute`` adds its image, plugins and nodes.

        Parameters
        ----------
        state
            Only the computes in this state, and then every one of them.
        history
            How many finished computes follow the live ones. 0 lists none.
        raw
            Answer with the daemon's resources whole. The environment's values and the plugins' credentials stay hidden.
        """
        return await self._ask(
            lambda client: computes(client, state, history),
            lambda found: tuple(map(_hidden, found)) if raw else _brief(tuple(_compute(compute, detail=False) for compute in found)),
        )

    async def get_compute(self, ref: str, raw: bool = False) -> str:
        """Read one compute: its state, what it asked for and what was bought, its cost, image and plugins, and each node's state, ssh address and last error.

        Parameters
        ----------
        ref
            The compute's id or name.
        raw
            Answer with the daemon's resource whole. The environment's values and the plugins' credentials stay hidden.
        """
        return await self._ask(
            lambda client: client.call("GET", f"/v1/computes/{ref}", v1.ComputeResource),
            lambda found: _hidden(found) if raw else _described(found),
        )

    async def list_tasks(
        self,
        compute: str,
        state: v1.TaskState | None = None,
        limit: int = 20,
        cursor: str | None = None,
        raw: bool = False,
    ) -> str:
        """List a compute's tasks, the function calls it was given: the running ones, then the queued, then the latest to finish.

        Each attempt at a task carries the error it failed with, and a failed function's traceback is in its ``details``.

        Parameters
        ----------
        compute
            The compute's id or name.
        state
            Only the tasks in this state.
        limit
            How many tasks to answer with.
        cursor
            The ``next_cursor`` of a previous answer, for the page after it.
        raw
            Answer with the daemon's resources whole, hashes and all.
        """
        return await self._ask(
            lambda client: client.call("GET", "/v1/tasks", v1.Page[v1.TaskResource], compute=compute, state=state, limit=limit, order="state", cursor=cursor),
            lambda page: page if raw else _paged(page, _as(TaskView)),
        )

    async def read_log(
        self,
        compute: str | None = None,
        task: str | None = None,
        node: str | None = None,
        types: tuple[str, ...] = (),
        contains: tuple[str, ...] = (),
        limit: int = 50,
        cursor: str | None = None,
        raw: bool = False,
    ) -> str:
        """Read the event log, newest first: what the nodes printed, tracebacks, task outcomes, and every change of state.

        An entry is when it was recorded, its type and what it said, without the compute, task or node the filters already name.

        Parameters
        ----------
        compute
            Only this compute's entries, by id or name.
        task
            Only this task's entries.
        node
            Only this node's entries, by id.
        types
            Only these events: ``node.console`` is what a node printed; others are ``task.failed``, ``task.succeeded``,
            ``node.lost``, ``node.failed``, ``compute.degraded``, ``compute.ready``.
        contains
            Only the entries whose line holds one of these strings, ignoring case, such as ``Traceback``.
        limit
            How many entries to answer with, at most 1000.
        cursor
            The ``next_cursor`` of a previous answer, for the older entries after it.
        raw
            Answer with the daemon's entries whole: sequence numbers, execution ids, and every field the filters name.
        """
        named = _named(compute=compute, task=task, node=node)
        return await self._ask(
            lambda client: client.call(
                "GET",
                "/v1/events/log",
                v1.Page[v1.LogEntryResource],
                compute=compute,
                task=task,
                node=node,
                types=types,
                contains=contains,
                limit=limit,
                cursor=cursor,
            ),
            lambda page: page if raw else _paged(page, lambda entry: _entry(entry, named)),
        )

    async def read_metrics(self, compute: str, node: tuple[str, ...] = (), name: tuple[str, ...] = (), raw: bool = False) -> str:
        """Read the newest value of each metric on each node of a compute: GPU utilisation, memory, temperature and power, CPU, memory, disk, network.

        Each node answers with ``at``, when its newest sample was taken, and ``values``, each metric's by name.

        Parameters
        ----------
        compute
            The compute's id or name.
        node
            Only these nodes, by id.
        name
            Only these metrics, such as ``gpu_util`` or ``cpu``.
        raw
            Answer with one sample per node and metric, each with the time it was taken, in milliseconds on the node's clock.
        """
        return await self._ask(
            lambda client: client.call("GET", f"/v1/computes/{compute}/metrics/latest", v1.Page[v1.MetricSampleResource], node=node, name=name),
            lambda page: page if raw else _brief(_latest(page)),
        )

    async def read_metric_history(
        self,
        compute: str,
        name: Annotated[tuple[str, ...], Field(min_length=1)],
        minutes: Positive = 60,
        until: AwareDatetime | None = None,
        step_seconds: Positive = 60,
        agg: v1.Aggregate = "avg",
        node: tuple[str, ...] = (),
    ) -> str:
        """Read how a compute's metrics moved over a window of time: one series per node and metric, one value per step.

        Each series pairs ``at``, milliseconds since the epoch on the node's clock, with ``values``.

        Parameters
        ----------
        compute
            The compute's id or name.
        name
            The metrics to read, at least one: ``gpu_util``, ``gpu_mem_mb``, ``gpu_temp_c``, ``gpu_power_w``, ``cpu``,
            ``mem_used_mb``, ``disk_used_pct``, ``net_rx_kbps``, ``net_tx_kbps``, or a custom metric's name.
        minutes
            How long the window is.
        until
            When the window ends, with its timezone. Unset, it ends now.
        step_seconds
            How long each value stands for. A window is answered in at most 2000 steps.
        agg
            How the samples of one step become its value.
        node
            Only these nodes, by id.
        """
        end = int(until.timestamp() * 1000) if until else time.time_ns() // 1_000_000
        return await self._ask(
            lambda client: client.call(
                "GET",
                f"/v1/computes/{compute}/metrics",
                v1.MetricHistoryResource,
                since=end - minutes * 60_000,
                until=end if until else None,
                step=step_seconds * 1000,
                agg=agg,
                node=node,
                name=name,
            ),
            _brief,
        )

    async def read_daemon_log(
        self,
        level: v1.LogLevel = "INFO",
        compute: str | None = None,
        node: str | None = None,
        component: tuple[str, ...] = (),
        contains: tuple[str, ...] = (),
        limit: int = 50,
        cursor: str | None = None,
        raw: bool = False,
    ) -> str:
        """Read the daemon's own log, newest first: how it reconciles, buys, connects and dispatches, and every failure with its exception.

        It is where a compute stuck in ``provisioning``, a node that never connected or a provider that keeps failing is explained.

        Parameters
        ----------
        level
            The lowest level kept.
        compute
            Only the lines about this compute, by id or name.
        node
            Only the lines about this node, by id.
        component
            Only these parts of the daemon: ``reconciler``, ``market``, ``machines``, ``connector``, ``ssh``, ``node``,
            ``dispatcher``, ``offers``, ``metering``, ``listeners``, ``http``.
        contains
            Only the lines that hold one of these strings, ignoring case, in the message or in its exception.
        limit
            How many lines to answer with, at most 1000.
        cursor
            The ``next_cursor`` of a previous answer, for the older lines after it.
        raw
            Answer with the daemon's lines whole: the call that logged each one, its logger and group, and every field the
            filters name.
        """
        named = _named(compute=compute, node=node)
        return await self._ask(
            lambda client: client.call(
                "GET",
                "/v1/daemon/log",
                v1.Page[v1.DaemonLogResource],
                level=level,
                compute=compute,
                node=node,
                component=component,
                contains=contains,
                limit=limit,
                cursor=cursor,
            ),
            lambda page: page if raw else _paged(page, lambda line: _without(_as(DaemonLineView)(line), named)),
        )

    async def list_offers(
        self,
        provider: str | None = None,
        accelerator: str | None = None,
        min_count: int | None = None,
        min_vram: float | None = None,
        max_price: float | None = None,
        spot: bool | None = None,
        limit: int = 20,
        refresh: bool = False,
        raw: bool = False,
    ) -> str:
        """List the machines the registered provider accounts offer, cheapest per accelerator first; ``total`` counts every match.

        Prices are in dollars per hour; ``billing_unit`` is what the time a machine is held is rounded up to.

        Parameters
        ----------
        provider
            Only this account, by id or name.
        accelerator
            Only this accelerator, in any spelling: ``A100``, ``H100``, ``RTX 4090``.
        min_count
            Least accelerators per machine.
        min_vram
            Least memory per accelerator, in GB.
        max_price
            Most an offer may cost, in dollars per hour.
        spot
            ``true`` lists only the offers that have a spot price.
        limit
            How many offers to answer with, at most 2000.
        refresh
            Fetch every provider's catalog again before answering, even one still fresh.
        raw
            Answer with the daemon's offers whole: ids, when each was fetched, and what is particular to its provider.
        """
        return await self._ask(
            lambda client: client.call(
                "GET",
                "/v1/offers",
                v1.Page[v1.OfferResource],
                provider=provider,
                accelerator=accelerator,
                min_count=min_count,
                min_vram=min_vram,
                max_price=max_price,
                spot=spot,
                limit=limit,
                refresh=refresh,
            ),
            lambda page: page if raw else _paged(page, _as(OfferView)),
        )

    async def list_providers(self, kinds: bool = False) -> str:
        """List the provider accounts the daemon holds, each with the last error it hit and when its offers were last fetched.

        Parameters
        ----------
        kinds
            List instead every provider kind this build supports, with the credentials each one reads from the environment.
        """
        if kinds:
            return await self._ask(lambda client: client.call("GET", "/v1/provider-kinds", tuple[v1.ProviderKindResource, ...]), _brief)
        return await self._ask(
            lambda client: client.call("GET", "/v1/providers", v1.Page[v1.ProviderResource]), lambda page: _paged(page, lambda account: account)
        )

    async def read_result(self, task: str) -> str:
        """Read what a task returned on each node that finished it, by rank, beside the task's ``state``.

        A file ``run_script`` ran whole returns its exit status. A ``sky.app`` function returns what it returned, as JSON, or
        ``exit`` when it exited instead. A task still running has no results yet, and an attempt that failed has none: its
        error is in ``list_tasks``. What a function returned is answered as it is, nulls and all.

        Parameters
        ----------
        task
            The task's id, from ``list_tasks``.
        """

        async def results(client: Client) -> tuple[v1.TaskResource, list[tuple[int, object]]]:
            found = await client.call("GET", f"/v1/tasks/{task}", v1.TaskResource)
            ranked = sorted(found.executions, key=lambda execution: execution.rank)
            blobs = [
                (execution.rank, blob)
                for execution in ranked
                if execution.result_sha256 and (blob := await client.blob(f"/v1/blobs/{execution.result_sha256}"))
            ]
            return found, [(rank, await _unpickled(task, blob)) for rank, blob in blobs]

        return await self._ask(results, lambda found: {"state": found[0].state, "results": [_returned(rank, value) for rank, value in found[1]]})

    async def create_compute(
        self,
        provider: str,
        name: str | None = None,
        accelerator: str | None = None,
        nodes: Count = 1,
        region: str | None = None,
        cpus: int | None = None,
        memory_gb: int | None = None,
        ttl: int | None = None,
        base: str | None = None,
        python: str | None = None,
        pip: tuple[str, ...] = (),
        apt: tuple[str, ...] = (),
        pip_indexes: tuple[str, ...] = (),
        env: Mapping[str, str] | None = None,
        mutable: bool = False,
        plugins: Mapping[str, Mapping[str, JsonValue]] | None = None,
    ) -> str:
        """Create a compute: buy ``nodes`` machines and keep them until the compute is deleted. It costs money from the moment they are bought.

        It answers at once, while the compute is still provisioning; ``get_compute`` follows it to ``ready``. When the daemon holds no
        account of the provider's kind, one is registered first, with the credentials this server's environment carries.

        Parameters
        ----------
        provider
            The provider kind: ``aws``, ``gcp``, ``runpod``, ``vastai``, ``lambda``, ``container`` (local Docker, no cost), or any other
            ``list_providers`` with ``kinds`` names.
        name
            A name to reach the compute by, besides its id.
        accelerator
            The accelerator each machine has: ``A100``, ``H100``, ``L4``. None asks for a machine without one.
        nodes
            How many machines.
        region
            Where to buy them.
        cpus
            Least vCPUs per machine.
        memory_gb
            Least memory per machine, in GB.
        ttl
            Seconds a machine may go with nobody connected before it removes itself. 0 never does.
        base
            The Docker image the machines start from.
        python
            The Python version the nodes run, such as ``3.12``.
        pip
            Packages to install, as pip spells them.
        apt
            apt packages to install.
        pip_indexes
            Extra package index URLs.
        env
            Environment variables on the nodes.
        mutable
            Let ``update_compute`` change the packages and the shipped code while the compute is up.
        plugins
            Plugins by kind, each with its parameters: ``{"torch": {"backend": "nccl"}, "huggingface": {}}``.
        """

        def plugin(kind: str, params: Mapping[str, JsonValue]) -> PluginRef:
            if kind not in PLUGINS:
                raise ToolError(f"unknown plugin '{kind}'; known: {', '.join(sorted(PLUGINS))}")
            try:
                return msgspec.convert(params, PLUGINS[kind], strict=False).ref()
            except msgspec.ValidationError as invalid:
                raise ToolError(f"plugin {kind}: {invalid}") from None

        if provider not in FACTORIES:
            raise ToolError(f"unknown provider '{provider}'; known: {', '.join(sorted(FACTORIES))}")

        account = FACTORIES[provider]()
        spec = compute_spec(
            account,
            accelerator=accelerator,
            nodes=bounds(nodes),
            region=region,
            cpus=cpus,
            memory_gb=memory_gb,
            ttl=ttl,
            image=Image(
                base=base,
                python=python,
                pip=pip,
                apt=apt,
                pip_indexes=tuple(PipIndex(url=index) for index in pip_indexes),
                env=dict(env or {}),
                mutable=mutable,
            ),
            plugins=tuple(plugin(kind, params) for kind, params in (plugins or {}).items()),
        )
        return await self._ask(lambda client: create(client, account, spec, name), _described)

    async def run_script(self, path: str, args: tuple[str, ...] = (), node: Literal["any", "all"] | int = "any") -> str:
        """Run a Python file on the compute it declares, as ``sky run`` does, and answer once the run's task is queued.

        The file declares its compute in a PEP 723 header (a ``# /// script`` block with ``[tool.skyward]``), and then all
        of it runs with ``args`` as its argv. Or it decorates functions with ``sky.app``, and then ``args`` name one and give
        its arguments as flags, such as ``["train", "--epochs", "5"]``; ``["--help"]`` answers with the functions and what
        they take, buying nothing.

        The answer is the compute the run stands on, usually still being bought, and ``task``, the run's task. A compute an
        earlier run of the same declaration left up is used again, and one another run still holds is refused.
        ``read_log`` with the task gives what it prints, and ``read_result`` its exit status or what its function returned
        once it settles. Nothing waits on the run: the daemon holds the compute for the task, whether or not this session
        goes on, and once the task settles deletes the compute unless ``delete_on_exit`` is false.

        Parameters
        ----------
        path
            The file, on the machine this server runs on.
        args
            The file's argv, or a function's name and its flags.
        node
            ``any`` runs it on one node, ``all`` on every node, a rank on that node.
        """
        file = Path(path).expanduser()
        if not file.is_file():
            raise ToolError(f"no such file: {file}")

        match _declared(file, args):
            case str(said):
                return said
            case Script() as script:
                return await self._ask(
                    lambda client: launch(client, script, node),
                    lambda launched: _brief(
                        {
                            "compute": _compute(launched.compute, detail=False),
                            "task": launched.task.id,
                            "delete_on_exit": script.app.delete_on_exit,
                        }
                    ),
                )

    async def scale_compute(self, ref: str, nodes: Count, max_nodes: Count | None = None) -> str:
        """Change how many machines a compute stands on. Growing it buys machines; shrinking it terminates them and whatever they held.

        ``nodes`` alone fixes the size. With ``max_nodes`` the compute grows and shrinks between the two with its load. The machines
        are bought or ended afterwards, so the answer is the compute with its new ``size``, not the finished resize.

        Parameters
        ----------
        ref
            The compute's id or name.
        nodes
            How many machines, or the fewest when ``max_nodes`` is given.
        max_nodes
            The most machines the compute grows to.
        """
        if max_nodes is not None and max_nodes < nodes:
            raise ToolError(f"max_nodes is the most the compute grows to, so it cannot be less than nodes ({max_nodes} < {nodes})")

        wanted = bounds(nodes if max_nodes is None else (nodes, max_nodes))
        return await self._ask(lambda client: writes.conditional(client, ref, "PATCH", msgspec.json.encode(ComputeSpecPatch(nodes=wanted))), _described)

    async def update_compute(
        self,
        ref: str,
        pip: tuple[str, ...] | None = None,
        pip_indexes: tuple[str, ...] | None = None,
        includes: tuple[str, ...] | None = None,
        excludes: tuple[str, ...] | None = None,
    ) -> str:
        """Change the packages or the local code a mutable compute's nodes run. A list given replaces the compute's; one left out keeps it.

        The nodes that are up bootstrap again with the new image and come back ready, and tasks submitted meanwhile wait for them.
        A compute created without ``mutable`` refuses with ``image_fixed``.

        Parameters
        ----------
        ref
            The compute's id or name.
        pip
            Packages to install, as pip spells them.
        pip_indexes
            Extra package index URLs.
        includes
            Local files and directories to pack and ship, relative to this server's working directory.
        excludes
            Name patterns left out of the includes.
        """
        return await self._ask(lambda client: update(client, ref, pip=pip, pip_index=pip_indexes, include=includes, exclude=excludes), _described)

    async def delete_compute(self, ref: str) -> str:
        """Delete a compute: terminate every machine it has, and with them whatever they held. Billing stops once they are gone.

        The delete is accepted, not done: the compute answers ``deleting`` until the provider confirms its machines are gone.

        Parameters
        ----------
        ref
            The compute's id or name.
        """
        return await self._ask(lambda client: writes.delete(client, ref), _described)

    async def cancel_task(self, task: str) -> str:
        """Cancel a task, by the id ``list_tasks`` gives it.

        A task no machine has started never runs. A running one is asked to stop, and its attempt is ``cancel_requested`` until the
        worker lets go. One that finished first keeps its outcome.

        Parameters
        ----------
        task
            The task's id.
        """
        return await self._ask(lambda client: writes.cancel(client, task), lambda found: _brief(_as(TaskView)(found)))

    async def exec_command(self, ref: str, command: str, node: Literal["all"] | int = "all") -> str:
        """Run a shell command on a compute's machines and answer, for each node, its exit code, stdout and stderr.

        It is the machine's shell, not the worker: for what the driver reports, what is on disk, what is running. It reaches a node
        whose worker is busy, and what it changes is gone with the machine.

        Parameters
        ----------
        ref
            The compute's id or name.
        command
            The command line, run by the node's shell.
        node
            ``all`` for every node, or one node's rank.
        """
        return await self._ask(lambda client: client.call("POST", f"/v1/computes/{ref}/exec", dict[str, Result], command=command, node=str(node)), _brief)

    async def _ask[T](self, work: Callable[[Client], Awaitable[T]], shown: Callable[[T], object]) -> str:
        """Run ``work`` against the daemon and answer with what ``shown`` makes of its result, as JSON, or with what refused it."""
        client = await Client.remote(self._url)
        try:
            return msgspec.json.encode(shown(await work(client))).decode()
        except httpx.ConnectError:
            raise ToolError(f"no daemon at {self._url}; start one with `sky server start`") from None
        except SkywardError as error:
            raise ToolError(f"{error.code}: {error.message}") from None
        except UnexpectedResponseError as error:
            raise ToolError(str(error)) from None
        finally:
            await client.close()


class OfferView(Struct, frozen=True, kw_only=True):
    provider_name: str
    kind: str
    instance_type: str
    accelerator: str | None
    accelerator_count: int
    vram: float | None
    cpus: int
    memory_gb: float
    disk_gb: float | None
    region: str | None
    architecture: v1.Architecture | None
    price: float | None
    spot_price: float | None
    on_demand_price: float | None
    billing_unit: v1.BillingUnit
    available: int | None


class NodeView(Struct, frozen=True, kw_only=True):
    id: str
    rank: int
    state: v1.NodeState
    market: v1.Market | None
    ssh: v1.Ssh | None
    busy: int
    progress: v1.Progress | None
    last_error: v1.Error | None


class ImageView(Struct, frozen=True, kw_only=True):
    base: str | None
    python: str | None
    pip: tuple[str, ...]
    apt: tuple[str, ...]
    pip_indexes: tuple[v1.PipIndex, ...]
    env: dict[str, str]
    includes: tuple[str, ...]
    excludes: tuple[str, ...]
    mutable: bool


class ComputeView(Struct, frozen=True, kw_only=True):
    """A compute as an assistant reads it: what it asked for (``specs``, ``size``), what was bought (``offer``), how it stands."""

    id: str
    name: str | None
    state: v1.ComputeState
    error: v1.Error | None
    placement: v1.Refusal | None
    specs: tuple[v1.Spec, ...]
    size: v1.NodeBounds
    offer: OfferView | None
    ready: int
    rate: float
    cost: float
    tasks: dict[str, int]
    ttl: int
    created_at: datetime
    ended: v1.Ending | None
    image: ImageView | None = None
    plugins: tuple[v1.PluginRef, ...] = ()
    nodes: tuple[NodeView, ...] = ()


class FunctionView(Struct, frozen=True, kw_only=True):
    name: str | None
    version: int | None


class AttemptView(Struct, frozen=True, kw_only=True):
    rank: int
    state: v1.ExecutionState
    node_id: str | None
    error: v1.Error | None
    started_at: datetime | None
    finished_at: datetime | None


class TaskView(Struct, frozen=True, kw_only=True):
    id: str
    function: FunctionView
    state: v1.TaskState
    dispatch: v1.Dispatch
    rank: int | None
    submitted_at: datetime
    finished_at: datetime | None
    queue_timeout_seconds: float | None
    run_timeout_seconds: float | None
    executions: tuple[AttemptView, ...]


class DaemonLineView(Struct, frozen=True, kw_only=True):
    at: datetime
    level: v1.LogLevel
    component: str | None
    compute: str | None
    node: str | None
    message: str
    fields: dict[str, str]
    exception: v1.LogException | None


class NodeMetricsView(Struct, frozen=True, kw_only=True):
    node: str
    at: datetime
    values: dict[str, float]


def _hidden(compute: v1.ComputeResource) -> v1.ComputeResource:
    """``compute`` with ``HIDDEN`` for the values of its environment and for its plugins' credentials.

    A plugin this build does not know has every parameter hidden, since nothing says which of them is a secret.
    """

    def plugin(ref: v1.PluginRef) -> v1.PluginRef:
        secret: Collection[str] = secrets(PLUGINS[ref.kind]) if ref.kind in PLUGINS else ref.params.keys()
        return replace(ref, params={name: HIDDEN if name in secret and value is not None else value for name, value in ref.params.items()})

    spec = compute.spec
    image = replace(spec.image, env=dict.fromkeys(spec.image.env, HIDDEN), shell_vars=dict.fromkeys(spec.image.shell_vars, HIDDEN))
    return replace(compute, spec=replace(spec, image=image, plugins=tuple(map(plugin, spec.plugins))))


def _compute(compute: v1.ComputeResource, *, detail: bool) -> ComputeView:
    """``compute`` as an assistant reads it; ``detail`` adds its image, plugins and nodes."""
    spec = _hidden(compute).spec
    return ComputeView(
        id=compute.id,
        name=compute.name,
        state=compute.status.state,
        error=compute.status.last_error,
        placement=compute.placement,
        specs=spec.specs,
        size=spec.nodes,
        offer=None if compute.offer is None else _as(OfferView)(compute.offer),
        ready=sum(node.state == "ready" for node in compute.nodes),
        rate=compute.rate,
        cost=compute.cost,
        tasks={state: count for state, count in msgspec.to_builtins(compute.tasks).items() if count},
        ttl=spec.ttl,
        created_at=compute.created_at,
        ended=compute.ended,
        image=_as(ImageView)(spec.image) if detail else None,
        plugins=spec.plugins if detail else (),
        nodes=tuple(map(_as(NodeView), compute.nodes)) if detail else (),
    )


def _described(compute: v1.ComputeResource) -> object:
    return _brief(_compute(compute, detail=True))


def _declared(file: Path, args: tuple[str, ...]) -> Script | str:
    """What ``sky run`` would run, or what it says instead when that is all ``args`` asked for: its help.

    The file is read here the way ``sky run`` reads it — imported, when its functions are what it declares — so a
    command line its function cannot take is refused before anything is bought. Whatever that prints is kept off
    stdout, which is this server's channel to its host.
    """
    said = io.StringIO()
    try:
        with redirect_stdout(said), redirect_stderr(said):
            return read(file, args)
    except SystemExit as ended:
        match ended.code:
            case 0 | None:
                return said.getvalue()
            case str(reason):
                raise ToolError(reason) from None
            case _:
                raise ToolError(said.getvalue().strip()) from None
    except Exception as failed:
        raise ToolError(f"{file} cannot be read: {type(failed).__name__}: {failed}") from None


async def _unpickled(task: str, blob: bytes) -> object:
    """What one attempt at ``task`` returned, loaded back, or why this process cannot: a class only the caller's code has."""
    try:
        return await codec.Pickle[object]().decode(blob)
    except Exception as failed:
        raise ToolError(f"task {task} returned a Python object this server cannot load ({type(failed).__name__}: {failed}); the SDK reads it") from None


def _returned(rank: int, value: object) -> dict[str, object]:
    """One rank's result: a file's exit status, a ``sky.app`` function's JSON, or any other value as JSON with ``str`` for what has none."""
    match value:
        case Returned(value=raw):
            return {"rank": rank, "returned": msgspec.json.decode(raw)}
        case Exited(status=status):
            return {"rank": rank, "exit": status}
        case _:
            return {"rank": rank, "returned": msgspec.to_builtins(value, enc_hook=str)}


def _entry(entry: v1.LogEntryResource, named: Collection[str]) -> dict[str, object]:
    """One event as one flat object: when it was recorded, its type, and what it said but for what the filters named."""
    left = {"at", "execution", *named}
    return {"at": entry.at, "type": entry.type} | {key: value for key, value in asdict(entry.data).items() if key not in left}


def _latest(page: v1.Page[v1.MetricSampleResource]) -> tuple[NodeMetricsView, ...]:
    """The newest samples, one object per node."""

    def measured(node: str, samples: tuple[v1.MetricSampleResource, ...]) -> NodeMetricsView:
        newest = max(sample.at for sample in samples)
        return NodeMetricsView(node=node, at=datetime.fromtimestamp(newest / 1000, UTC), values={sample.name: sample.value for sample in samples})

    ordered = sorted(page.items, key=lambda sample: sample.node)
    return tuple(measured(node, tuple(samples)) for node, samples in groupby(ordered, key=lambda sample: sample.node))


def _named(**filters: str | None) -> frozenset[str]:
    """The fields a reading's filters fix, which every line it answers with would only repeat."""
    return frozenset(field for field, value in filters.items() if value)


def _without(view: Struct, named: Collection[str]) -> dict[str, object]:
    return {field: value for field, value in asdict(view).items() if field not in named}


def _as[V](view: type[V]) -> Callable[[object], V]:
    """A resource as ``view``, whose fields are some of the resource's, by the same names."""
    return lambda found: msgspec.convert(found, view, from_attributes=True)


def _paged[T](page: v1.Page[T], view: Callable[[T], object]) -> dict[str, object]:
    """A page as an answer: ``items`` even when there are none, each item brief, and the cursor and total when there are."""
    rest = {key: value for key, value in (("next_cursor", page.next_cursor), ("total", page.total)) if value is not None}
    return {"items": [_brief(view(item)) for item in page.items], **rest}


def _brief(answer: object) -> object:
    """``answer`` as JSON values, without a field that is null or empty at any depth, and with every time to the second."""

    def pruned(value: object) -> object:
        match value:
            case dict():
                kept = ((key, pruned(item)) for key, item in value.items())
                return {key: item for key, item in kept if item not in (None, "", [], {})}
            case list() | tuple():
                return [pruned(item) for item in value]
            case datetime():
                return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
            case _:
                return value

    return pruned(msgspec.to_builtins(answer, builtin_types=(datetime,)))


__all__ = ["HIDDEN", "INSTRUCTIONS", "Tools", "server"]
