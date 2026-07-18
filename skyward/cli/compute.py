"""``sky compute`` — the computes endpoint, as a command.

Every command here is one or two HTTP calls and a table. What a compute *is*
lives in the daemon; the flags only say which one, and creating one is a
``ComputeSpec`` posted whole rather than a decision taken locally.

Credentials are the exception, and only because they have to be: the daemon
never reads the environment, so the provider row is registered from this
process — the same thing the SDK does before it creates a pool.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import msgspec
from cyclopts import Parameter

from skyward.cli import compute_app
from skyward.cli._client import Work, call
from skyward.cli._output import Output, render
from skyward.protocol.schemas import Compute, ComputeCreate, ComputeSpec, Node, NodeBounds, Page, ProviderCreate, ProviderRef, Spec
from skyward.sdk import provider as factories
from skyward.sdk.client import Client
from skyward.sdk.errors import SkywardError
from skyward.sdk.provider import Provider

COMPUTE_COLUMNS = ("id", "name", "state", "ready", "total", "generation", "created")
NODE_COLUMNS = ("id", "rank", "state", "desired", "machine", "address", "accelerator", "$/h")

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
    url: Annotated[str | None, Parameter(help="Daemon URL")] = None,
    database: Annotated[Path | None, Parameter(help="Embedded daemon database")] = None,
    output: Annotated[Output, Parameter(help="table or json")] = "table",
) -> None:
    """Create a compute and return without waiting for it to be ready."""
    from skyward.protocol.accelerators import resolve

    if provider not in FACTORIES:
        raise SystemExit(f"unknown provider '{provider}'; known: {', '.join(sorted(FACTORIES))}")

    account = FACTORIES[provider]()
    spec = ComputeSpec(
        specs=(
            Spec(
                provider=ProviderRef(kind=account.kind, config=dict(account.config)),
                accelerator=resolve(accelerator, None)[0],
                cpus=cpus,
                memory_gb=memory,
                region=region,
            ),
        ),
        nodes=NodeBounds(desired=nodes),
    )

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

    async def work(client: Client) -> Compute:
        current = await client.call("GET", f"/v1/computes/{ref}", Compute)
        return await client.call(
            "DELETE",
            f"/v1/computes/{current.id}",
            Compute,
            headers={"If-Match": f'"{current.revision}"', "Idempotency-Key": uuid.uuid4().hex},
        )

    render(COMPUTE_COLUMNS, [_compute_row(_call(work, url=url, database=database))], output=output)


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
    try:
        await client.call("GET", f"/v1/providers/{account.name}", dict[str, object])
    except SkywardError as error:
        if error.code != "not_found":
            raise
        await client.call(
            "POST",
            "/v1/providers",
            dict[str, object],
            body=msgspec.json.encode(
                ProviderCreate(
                    name=account.name,
                    kind=account.kind,
                    credentials=dict(account.credentials),
                    config=dict(account.config),
                ),
            ),
        )


def _call[T](work: Work[T], *, url: str | None = None, database: Path | None = None) -> T:
    """Run the work, and turn a refusal into a message rather than a traceback."""
    try:
        return call(work, url=url, database=database)
    except SkywardError as error:
        raise SystemExit(f"{error.code}: {error.message}") from None


__all__ = ["create_compute", "delete_compute", "get_compute", "list_computes", "view_compute"]
