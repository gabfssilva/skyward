from litestar import Controller, delete, get, post

from skyward.providers import registry
from skyward.server.application import ports
from skyward.shared.schemas import Page, Provider, ProviderCreate, ProviderKind


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
    async def list(self) -> tuple[ProviderKind, ...]:
        return registry.kinds()


class ProviderController(Controller):
    path = "/providers"
    tags = ["providers"]

    @get(summary="List registered providers")
    async def list(self, providers: ports.Providers) -> Page[Provider]:
        return await providers.list()

    @post(
        status_code=201,
        summary="Register a provider account",
        description=(
            "A provider is a named account, not a kind: two AWS accounts, two Vast keys and two regions can coexist, and "
            "a compute refers to the provider it wants by id or name.\n\n"
            "Credentials are validated against the kind's `credential_fields` before the row is written, and are never "
            "returned by any read path."
        ),
    )
    async def create(self, data: ProviderCreate, providers: ports.Providers) -> Provider:
        return await providers.create(data)

    @get("/{provider_id:str}", summary="Read a provider", description="Accepts an id or a name. Credentials are never included.")
    async def read(self, provider_id: str, providers: ports.Providers) -> Provider:
        return await providers.get(provider_id)

    @delete(
        "/{provider_id:str}",
        status_code=204,
        summary="Remove a provider",
        description="Drops the account and its cached offers. Computes already running on it are not touched.",
    )
    async def destroy(self, provider_id: str, providers: ports.Providers) -> None:
        await providers.delete(provider_id)
