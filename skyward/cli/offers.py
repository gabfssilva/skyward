"""sky offers — browse GPU offers and pricing."""

from __future__ import annotations

import asyncio
from typing import Annotated

from cyclopts import Parameter

from . import offers_app
from ._output import console, format_price, print_table

_EMPTY_HINT = (
    "[dim]No offers in local catalog. "
    "Run [bold]sky offers fetch[/bold] to populate it.[/dim]"
)


def _format_regions(regions: str | None, count: int) -> str:
    if not regions:
        return "-"
    if count <= 3:
        return regions
    first_two = regions.split(",", 2)[:2]
    return f"{', '.join(first_two)} +{count - 2}"


def _format_accelerator(name: str, count: float) -> str:
    if not name:
        return "-"
    n = int(count) if count == int(count) else count
    return name if n == 1 else f"{n}x {name}"


_SPECIFIC_SKIP_KEYS = frozenset({
    "id", "region", "architecture", "vcpus", "memory_gb", "gpu_count",
    "gpu_vram_gb", "ram_gb", "hourly_rate",
    "flavor_name", "image_name", "os_image",
    "inet_down", "inet_up", "disk_bw", "driver", "verification",
})


def _format_specific(raw: str | None, max_len: int = 80) -> str:
    """Format the specific JSON blob as compact key=value pairs."""
    if not raw:
        return "-"
    import json as _json

    try:
        data = _json.loads(raw)
    except (ValueError, TypeError):
        return "-"
    if not isinstance(data, dict):
        return str(data)[:max_len]
    parts = [
        f"{k}={v}" for k, v in data.items()
        if k not in _SPECIFIC_SKIP_KEYS
    ]
    if not parts:
        return "-"
    return ", ".join(parts)


def _load_repo(*, with_providers: bool = False) -> object:
    from skyward.offers.repository import OfferRepository

    async def _create() -> OfferRepository:
        providers = None
        if with_providers:
            providers = await _configured_providers()
        return await OfferRepository.create(providers=providers)

    return asyncio.run(_create())


async def _configured_providers() -> list[object]:
    from skyward.cli.providers import _PROVIDER_CREDENTIALS, _check_credentials
    from skyward.config import _get_provider_map

    provider_map = _get_provider_map()
    providers = []
    for name in sorted(_PROVIDER_CREDENTIALS):
        if _check_credentials(name)[0] != "ok":
            continue
        cls = provider_map.get(name)
        if cls is None:
            continue
        try:
            providers.append(await cls().create_provider())
        except Exception:
            continue
    return providers


def _resolve_provider_instances(names: list[str]) -> list[object]:
    from skyward.config import _get_provider_map

    provider_map = _get_provider_map()
    instances = []
    for name in names:
        cls = provider_map.get(name)
        if cls is None:
            console.print(f"[yellow]Unknown provider '{name}', skipping[/yellow]")
            continue
        instances.append(cls())
    return instances


@offers_app.command(name="fetch")
def fetch_offers(
    *,
    provider: Annotated[list[str] | None, Parameter(name="--provider", help="Providers to fetch (repeatable). Default: all configured")] = None,
) -> None:
    """Fetch live offers from providers and cache locally."""
    from skyward.cli.providers import _PROVIDER_CREDENTIALS, _check_credentials

    names = provider or [p for p in sorted(_PROVIDER_CREDENTIALS) if _check_credentials(p)[0] == "ok"]

    if not names:
        console.print("[red]No configured providers found. Set API keys first.[/red]")
        return

    console.print(f"Fetching offers from: {', '.join(names)}")
    configs = _resolve_provider_instances(names)

    if not configs:
        console.print("[red]No valid providers to fetch from.[/red]")
        return

    async def _fetch() -> int:
        from skyward.offers.repository import OfferRepository

        providers_ready: list[tuple[str, object]] = []
        for cfg in configs:
            provider_name = cfg.type  # type: ignore[union-attr]
            try:
                p = await cfg.create_provider()  # type: ignore[union-attr]
                console.print(f"  [green]ok[/green] {provider_name}", end="")
                providers_ready.append((provider_name, p))
            except Exception as exc:
                console.print(f"  [red]fail[/red] {provider_name}: {exc!s:.80}")

        if not providers_ready:
            return 0

        provider_instances = [p for _, p in providers_ready]
        repo = await OfferRepository.create(providers=provider_instances)
        total = await repo.fetch_all()

        for name, _ in providers_ready:
            count = len(repo.query(f"SELECT * FROM catalog WHERE provider = '{name}'"))
            console.print(f"  ({count} offers)")

        repo.close()
        return total

    total = asyncio.run(_fetch())
    console.print(f"\n[bold]{total}[/bold] offers cached.")


@offers_app.command(name="list")
def list_offers(
    *,
    provider: Annotated[list[str] | None, Parameter(name="--provider", help="Filter by provider (repeatable)")] = None,
    accelerator: Annotated[str | None, Parameter(name="--accelerator", help="Filter by GPU name")] = None,
    gpus: Annotated[float | None, Parameter(name="--gpus", help="Filter by accelerator count")] = None,
    vram: Annotated[float | None, Parameter(name="--vram", help="Minimum VRAM in GB")] = None,
    spot: Annotated[bool, Parameter(name="--spot", help="Only spot offers")] = False,
    sort: Annotated[str, Parameter(name="--sort", help="Sort by: price, vram, gpus")] = "price",
    limit: Annotated[int, Parameter(name="--limit", help="Max rows to display")] = 20,
    json: Annotated[bool, Parameter(name="--json", help="JSON output")] = False,
) -> None:
    """List GPU offers across providers."""
    from skyward.offers.repository import OfferRepository

    repo: OfferRepository = _load_repo(with_providers=True)  # type: ignore[assignment]

    stale = repo.stale_providers()
    if stale and not json:
        console.print(f"[dim]Refreshing {', '.join(stale)}...[/dim]")
    if stale:
        asyncio.run(repo.refresh())

    query = repo.select()

    if provider:
        for p in provider:
            query = query.provider(p)
    if accelerator:
        query = query.accelerator(accelerator)
    if gpus is not None:
        query = query.accelerator_count(gpus)
    if vram is not None:
        query = query.accelerator_memory(vram)
    if spot:
        query = query.spot()

    match sort:
        case "price":
            order = "COALESCE(MIN(spot_price), MIN(on_demand_price)) ASC NULLS LAST"
        case "vram":
            order = "accelerator_memory_gb DESC"
        case "gpus":
            order = "accelerator_count DESC"
        case _:
            order = "COALESCE(MIN(spot_price), MIN(on_demand_price)) ASC NULLS LAST"

    where = " AND ".join(query._clauses) if query._clauses else "1=1"
    params = tuple(query._params)

    grouped_sql = f"""\
SELECT
    provider, accelerator_name, accelerator_count,
    accelerator_memory_gb, vcpus, memory_gb,
    MIN(spot_price) AS spot_price,
    MIN(on_demand_price) AS on_demand_price,
    GROUP_CONCAT(DISTINCT region) AS regions,
    COUNT(*) AS region_count,
    MAX(specific) AS specific
FROM catalog
WHERE {where}
GROUP BY provider, accelerator_name, accelerator_count,
         accelerator_memory_gb, vcpus, memory_gb
ORDER BY {order}
"""

    count_sql = f"SELECT COUNT(*) FROM ({grouped_sql})"
    total = repo._db.execute(count_sql, params).fetchone()[0]

    grouped_sql += f" LIMIT {limit}"
    raw_rows = repo._db.execute(grouped_sql, params).fetchall()
    repo.close()

    if not raw_rows and not json:
        console.print(_EMPTY_HINT)
        return

    columns = ["Provider", "Accelerator", "VRAM", "vCPUs", "RAM", "Spot", "On-Demand", "Regions", "Details"]
    rows = [
        (
            r["provider"],
            _format_accelerator(r["accelerator_name"], r["accelerator_count"]),
            f"{r['accelerator_memory_gb']:.0f}GB" if r["accelerator_memory_gb"] else "-",
            int(r["vcpus"]) if r["vcpus"] == int(r["vcpus"]) else r["vcpus"],
            f"{r['memory_gb']:.0f}GB",
            format_price(r["spot_price"]),
            format_price(r["on_demand_price"]),
            _format_regions(r["regions"], r["region_count"]),
            _format_specific(r["specific"]),
        )
        for r in raw_rows
    ]

    print_table(columns, rows, as_json=json)

    if not json and total > len(raw_rows):
        console.print(f"[dim]Showing {len(raw_rows)} of {total} offers. Use --limit to see more.[/dim]")


@offers_app.command(name="query")
def query_offers(
    sql: Annotated[str, Parameter(help="Raw SQL against the catalog VIEW")],
    *,
    json: Annotated[bool, Parameter(name="--json", help="JSON output")] = False,
) -> None:
    """Execute raw SQL against the offer catalog."""
    from skyward.offers.repository import OfferRepository

    repo: OfferRepository = _load_repo()  # type: ignore[assignment]
    results = repo.query(sql)
    repo.close()

    if not results:
        print_table([], [], as_json=json)
        return

    fields = [f.name for f in results[0].__dataclass_fields__.values()]
    rows = [tuple(getattr(r, f) for f in fields) for r in results]
    print_table(fields, rows, as_json=json)


@offers_app.command(name="summary")
def summary_offers(
    *,
    accelerator: Annotated[str | None, Parameter(name="--accelerator", help="Filter by GPU name")] = None,
    json: Annotated[bool, Parameter(name="--json", help="JSON output")] = False,
) -> None:
    """Show average pricing per GPU per provider."""
    from skyward.offers.repository import OfferRepository

    repo: OfferRepository = _load_repo()  # type: ignore[assignment]
    summary_rows = repo.summary(accelerator=accelerator)
    repo.close()

    columns = ["GPU", "Provider", "Offers", "Avg Spot", "Avg On-Demand"]
    rows = [
        (
            row["gpu"],
            row["provider"],
            row["offers"],
            format_price(row["avg_spot"]),
            format_price(row["avg_on_demand"]),
        )
        for row in summary_rows
    ]

    print_table(columns, rows, as_json=json)
