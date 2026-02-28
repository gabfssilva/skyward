"""Thunder Compute provider for Skyward.

Thunder Compute is a cloud provider offering GPU instances with CUDA support.
Supports production mode (full CUDA + multi-GPU) and prototyping mode (cheaper limited CUDA).

NOTE: Only config classes are imported at package level to avoid deps.
For handlers and modules, import explicitly:

    from skyward.providers.thunder.provider import ThunderProvider
    from skyward.providers.thunder.client import ThunderClient

Environment Variables:
    TNR_API_TOKEN: API token (falls back to ~/.thunder/token if not set)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .client import ThunderClient, ThunderError
    from .provider import ThunderProvider

from .config import ThunderCompute

__all__ = [
    "ThunderCompute",
    "ThunderClient",
    "ThunderError",
    "ThunderProvider",
]
