"""RunPod provider configuration.

Immutable configuration dataclass for RunPod provider.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from skyward.api.provider import ProviderConfig

if typing.TYPE_CHECKING:
    from skyward.providers.runpod.provider import RunPodProvider

# =============================================================================
# Cloud Type Enum
# =============================================================================


class CloudType(Enum):
    """RunPod cloud type options."""

    SECURE = "secure"
    COMMUNITY = "community"


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class RunPod(ProviderConfig):
    """RunPod GPU Pods provider configuration.

    SSH keys are automatically detected from ~/.ssh/id_ed25519.pub or
    ~/.ssh/id_rsa.pub.

    Features:
        - Auto data center: If not specified, RunPod selects best location.

    Example:
        >>> from skyward.providers.runpod import RunPod
        >>> config = RunPod(data_center_ids=("EU-RO-1",))

    Args:
        api_key: RunPod API key. Falls back to RUNPOD_API_KEY env var.
        cloud_type: Cloud type (SECURE or COMMUNITY). Default: SECURE.
        container_disk_gb: Container disk size in GB. Default: 50.
        volume_gb: Persistent volume size in GB. Default: 20.
        volume_mount_path: Volume mount path. Default: /workspace.
        data_center_ids: Preferred data center IDs or "global" for auto-selection.
        ports: Port mappings (e.g., ["22/tcp", "8888/http"]). Default: ["22/tcp"].
        provision_timeout: Instance provision timeout in seconds. Default: 300.
        bootstrap_timeout: Bootstrap timeout in seconds. Default: 600.
        instance_timeout: Auto-shutdown in seconds (safety timeout). Default: 300.
        registry_auth: Name of the container registry credential registered in RunPod
            account settings. Authenticates Docker Hub pulls to avoid rate limits.
            Set to None to skip. Default: "docker hub".
    """

    api_key: str | None = None
    cloud_type: CloudType = CloudType.SECURE
    ubuntu: Literal["20.04", "22.04", "24.04", "newest"] | str = "newest"
    container_disk_gb: int = 50
    volume_gb: int = 20
    volume_mount_path: str = "/workspace"
    data_center_ids: tuple[str, ...] | Literal["global"] = "global"
    ports: tuple[str, ...] = ("22/tcp",)
    provision_timeout: float = 300.0
    bootstrap_timeout: float = 600.0
    instance_timeout: int = 300
    request_timeout: int = 30
    cpu_clock: Literal["3c", "5c"] | str = "3c"
    bid_multiplier: float = 1
    registry_auth: str | None = "docker hub"

    async def create_provider(self) -> RunPodProvider:
        from skyward.providers.runpod.provider import RunPodProvider
        return await RunPodProvider.create(self)

    @property
    def type(self) -> str: return "runpod"

    @property
    def region(self) -> str:
        if self.data_center_ids == "global":
            return "global"
        return self.data_center_ids[0]
