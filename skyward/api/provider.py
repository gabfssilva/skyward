from typing import Protocol, runtime_checkable


@runtime_checkable
class ProviderConfig[P](Protocol):
    async def create_provider(self) -> P: ...
