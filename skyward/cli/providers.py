"""sky providers — the accounts this daemon can provision on.

A provider is a registered account, not a kind: the daemon holds the
credentials and is the only thing that ever tries them. So checking one is a
read, not a probe — the row already carries the last error the daemon hit and
when it last managed to fetch offers, which is the answer the CLI would have
gone looking for.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from cyclopts import Parameter

from skyward.shared.schemas import Page, Provider, ProviderKind

from . import providers_app
from ._client import call
from ._output import Output, render

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
    """
    if kinds:
        supported = call(lambda client: client.call("GET", "/v1/provider-kinds", tuple[ProviderKind, ...]), url=url)
        render(KINDS, [(kind.kind, ", ".join(kind.credential_fields), kind.offers_ttl_seconds) for kind in supported], output=output)
        return

    page = call(lambda client: client.call("GET", "/v1/providers", Page[Provider]), url=url)
    render(REGISTERED, [(p.name, p.kind, p.id, p.offers_count, p.offers_fetched_at) for p in page.items], output=output)


@providers_app.command(name="check")
def check_providers(
    name: Annotated[str | None, Parameter(help="A provider id or name. Omit to check every registered account.")] = None,
    *,
    output: Output = "table",
    url: str | None = None,
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
    """
    providers = call(lambda client: _read(client, name), url=url)
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


__all__ = ["check_providers", "list_providers"]
