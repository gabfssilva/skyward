from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .client import LambdaClient, LambdaError
    from .provider import LambdaCloudProvider

from .config import LambdaCloud

__all__ = ["LambdaCloud", "LambdaClient", "LambdaCloudProvider", "LambdaError"]


def __getattr__(name: str) -> Any:
    if name in ("LambdaClient", "LambdaError"):
        from .client import LambdaClient, LambdaError

        return LambdaClient if name == "LambdaClient" else LambdaError
    if name == "LambdaCloudProvider":
        from .provider import LambdaCloudProvider

        return LambdaCloudProvider
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
