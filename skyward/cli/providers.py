"""sky providers — the accounts this daemon can provision on.

A provider is a registered account, not a kind: the daemon holds the
credentials and is the only thing that ever tries them. So checking one is a
read, not a probe — the row already carries the last error the daemon hit and
when it last managed to fetch offers, which is the answer the CLI would have
gone looking for.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import msgspec
from cyclopts import Parameter

from skyward.core.errors import SkywardError
from skyward.core.provider import resolve
from skyward.shared.schemas import Page, Provider, ProviderCreate, ProviderKind

from . import providers_app
from ._client import call
from ._output import Output, render
from .compute import FACTORIES

if TYPE_CHECKING:
    from skyward.core.client import Client

KINDS = ("kind", "credentials", "offers ttl")
REGISTERED = ("name", "kind", "id", "offers", "fetched")
CHECKS = ("name", "kind", "status", "offers", "fetched", "detail")


@providers_app.command(name="list")
def list_providers(
    *,
    kinds: Annotated[bool, Parameter(help="List the kinds this build supports instead of the registered accounts.")] = False,
    output: Output = "table",
    url: str | None = None,
    database: Path | None = None,
) -> None:
    """List the registered provider accounts.

    Parameters
    ----------
    kinds
        Show what can be registered — every kind, the credentials it needs and
        how long its offers stay fresh — rather than what is.
    output
        ``table`` for a person, ``json`` for a program.
    url
        The daemon to ask. Defaults to ``SKYWARD_URL``, else an embedded one.
    database
        Where the embedded daemon keeps its state.
    """
    if kinds:
        supported = call(lambda client: client.call("GET", "/v1/provider-kinds", tuple[ProviderKind, ...]), url=url, database=database)
        render(KINDS, [(kind.kind, ", ".join(kind.credential_fields), kind.offers_ttl_seconds) for kind in supported], output=output)
        return

    page = call(lambda client: client.call("GET", "/v1/providers", Page[Provider]), url=url, database=database)
    render(REGISTERED, [(p.name, p.kind, p.id, p.offers_count, p.offers_fetched_at) for p in page.items], output=output)


@providers_app.command(name="check")
def check_providers(
    name: Annotated[str | None, Parameter(help="A provider id or name. Omit to check every registered account.")] = None,
    *,
    output: Output = "table",
    url: str | None = None,
    database: Path | None = None,
) -> None:
    """Report whether the daemon's last use of each account worked.

    Parameters
    ----------
    name
        The account to check, by id or by name.
    output
        ``table`` for a person, ``json`` for a program.
    url
        The daemon to ask. Defaults to ``SKYWARD_URL``, else an embedded one.
    database
        Where the embedded daemon keeps its state.
    """
    providers = call(lambda client: _read(client, name), url=url, database=database)
    render(CHECKS, [_check(provider) for provider in providers], output=output)


async def _read(client: Client, name: str | None) -> tuple[Provider, ...]:
    if name:
        return (await client.call("GET", f"/v1/providers/{name}", Provider),)
    return (await client.call("GET", "/v1/providers", Page[Provider])).items


def _check(provider: Provider) -> tuple[object, ...]:
    status, detail = _verdict(provider)
    return (provider.name, provider.kind, status, provider.offers_count, provider.offers_fetched_at, detail)


def _verdict(provider: Provider) -> tuple[str, str | None]:
    match (provider.last_error, provider.offers_fetched_at):
        case (None, None):
            return ("unused", "credentials never exercised")
        case (None, _):
            return ("ok", None)
        case (error, _):
            return ("error", error.message)


@providers_app.command(name="set")
def set_provider(
    kind: str,
    *,
    config: Annotated[list[str] | None, Parameter(help="A setting, as key=value. Repeat for more than one.")] = None,
    name: Annotated[str | None, Parameter(help="Register it under this name. Defaults to the kind.")] = None,
    output: Output = "table",
    url: str | None = None,
    database: Path | None = None,
) -> None:
    """Register an account, or change the settings of the one already registered.

    The settings are the account's own fields — ``cloud_type=community`` for
    RunPod, ``region=eu-west-1`` for AWS — and they are checked against the
    account before anything is sent, so a value the provider has no name for is
    refused here rather than at provisioning time.

    The credentials are not written here and cannot be: they are read from the
    environment, the same way a pool reads them, so a key never sits in a shell
    history. The row is what the daemon builds its adapter from, which is why
    this exists — a compute names a kind, and the settings behind it live here.

    Parameters
    ----------
    kind
        Which provider: ``runpod``, ``aws``, ``container``, and so on.
    config
        Settings, as ``key=value``. Repeatable.
    name
        Register it under this name. Defaults to the kind, which is what a
        compute created without a name looks for.
    output
        ``table`` for a person, ``json`` for a program.
    url
        The daemon to ask. Defaults to ``SKYWARD_URL``, else an embedded one.
    database
        Where the embedded daemon keeps its state.
    """
    if kind not in FACTORIES:
        raise SystemExit(f"unknown provider '{kind}'; known: {', '.join(sorted(FACTORIES))}")

    try:
        written = msgspec.convert(_settings(config or []), FACTORIES[kind]().__class__, strict=False)
    except msgspec.ValidationError as refused:
        raise SystemExit(f"{kind}: {refused}") from None

    credentials, settings = resolve(written)
    body = ProviderCreate(name=name or kind, kind=kind, credentials=credentials, config=settings)
    registered = call(lambda client: _upsert(client, body), url=url, database=database)

    render(REGISTERED, [(registered.name, registered.kind, registered.id, registered.offers_count, registered.offers_fetched_at)], output=output)


def _settings(pairs: list[str]) -> dict[str, str]:
    """``key=value`` as a mapping, refusing anything that is not one."""
    written: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise SystemExit(f"--config takes key=value, not {pair!r}")
        written[key] = value
    return written


async def _upsert(client: Client, body: ProviderCreate) -> Provider:
    """Write the account, whether or not it is already there."""
    try:
        await client.call("GET", f"/v1/providers/{body.name}", Provider)
    except SkywardError as error:
        if error.code != "not_found":
            raise
        return await client.call("POST", "/v1/providers", Provider, body=msgspec.json.encode(body))

    return await client.call("PUT", f"/v1/providers/{body.name}", Provider, body=msgspec.json.encode(body))


__all__ = ["check_providers", "list_providers", "set_provider"]
