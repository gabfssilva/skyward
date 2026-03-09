"""Provider configuration protocol.

Defines the ``ProviderConfig[P]`` protocol that every cloud provider
config (``sky.AWS()``, ``sky.VastAI()``, etc.) must satisfy.  Config
objects are lightweight — no SDK imports at module level.  The actual
provider implementation is created lazily via ``create_provider()``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ProviderConfig[P](Protocol):
    """Configuration for a cloud provider.

    Lightweight config object — no SDK imports at module level.
    The actual provider implementation is created lazily via
    ``create_provider()``.

    Every provider config (``sky.AWS()``, ``sky.VastAI()``, etc.)
    implements this protocol.

    Examples
    --------
    >>> config = sky.AWS(region="us-east-1")
    >>> config.type
    'aws'
    """

    @property
    def type(self) -> str:
        """Provider identifier string (e.g., ``"aws"``, ``"vastai"``)."""
        ...

    async def create_provider(self) -> P:
        """Instantiate the provider implementation.

        Lazily import the provider SDK and construct the provider object.
        Called once during pool startup.

        Returns
        -------
        P
            The provider implementation.
        """
        ...
