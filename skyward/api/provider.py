from typing import Protocol, runtime_checkable


@runtime_checkable
class ProviderConfig[P](Protocol):
    @property
    def type(self) -> str: ...

    async def create_provider(self) -> P: ...
