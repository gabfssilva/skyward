from litestar import Controller, delete, get, post, put

from skyward.api import v1
from skyward.providers import registry
from skyward.server.application import ports
from skyward.server.http.exceptions import failures
from skyward.server.http.representation import recast
from skyward.shared.schemas import ProviderCreate


class ProviderKindController(Controller):
    path = "/provider-kinds"
    tags = ["providers"]

    @get(
        summary="List the provider kinds this build supports",
        description=(
            "Capability negotiation happens here, before anything is created: each kind declares the credentials it "
            "needs and how long its offers stay fresh. A kind absent from this list cannot be registered."
        ),
    )
    async def list(self) -> tuple[v1.ProviderKindResource, ...]:
        return recast(registry.kinds(), tuple[v1.ProviderKindResource, ...])


class ProviderController(Controller):
    path = "/providers"
    tags = ["providers"]

    @get(
        summary="List registered providers",
        description="The accounts this daemon can buy machines from. Credentials are not among the fields returned.",
    )
    async def list(self, providers: ports.Providers) -> v1.Page[v1.ProviderResource]:
        return recast(await providers.list(), v1.Page[v1.ProviderResource])

    @post(
        status_code=201,
        summary="Register a provider account",
        description=(
            "A provider is a named account, not a kind: two AWS accounts, two Vast keys and two regions can coexist, and "
            "a compute refers to the provider it wants by id or name.\n\n"
            "Credentials are validated against the kind's `credential_fields` before the row is written, and are never "
            "returned by any read path."
        ),
        responses=failures(409, 422),
    )
    async def create(self, data: v1.CreateProviderResource, providers: ports.Providers) -> v1.ProviderResource:
        return recast(await providers.create(recast(data, ProviderCreate)), v1.ProviderResource)

    @get(
        "/{provider_id:str}",
        summary="Read a provider",
        description="Accepts an id or a name. Credentials are never included.",
        responses=failures(404),
    )
    async def read(self, provider_id: str, providers: ports.Providers) -> v1.ProviderResource:
        return recast(await providers.get(provider_id), v1.ProviderResource)

    @put(
        "/{provider_id:str}",
        summary="Update a provider account",
        description=(
            "Replaces the account wholesale, credentials included — there is no partial update, because a half-written "
            "credential set is one that fails at launch rather than here. The cached offers are dropped with it."
        ),
        responses=failures(404, 409, 422),
    )
    async def update(self, provider_id: str, data: v1.CreateProviderResource, providers: ports.Providers) -> v1.ProviderResource:
        return recast(await providers.update(provider_id, recast(data, ProviderCreate)), v1.ProviderResource)

    @delete(
        "/{provider_id:str}",
        status_code=204,
        summary="Remove a provider",
        description="Drops the account and its cached offers. Computes already running on it are not touched.",
        responses=failures(404),
    )
    async def destroy(self, provider_id: str, providers: ports.Providers) -> None:
        await providers.delete(provider_id)
