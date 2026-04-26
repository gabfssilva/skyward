"""RunPod provider configuration.

Immutable configuration dataclass for RunPod provider.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import Literal

from skyward.containers import DockerImage
from skyward.core.provider import ProviderConfig

type ClusterMode = Literal["instant", "individual"]

if typing.TYPE_CHECKING:
    from skyward.providers.runpod.provider import RunPodProvider

# =============================================================================
# Configuration
# =============================================================================

type Datacenter = Literal[
    'EU-RO-1',
    'CA-MTL-1',
    'EU-SE-1',
    'US-IL-1',
    'EUR-IS-1',
    'EU-CZ-1',
    'US-TX-3',
    'EUR-IS-2',
    'US-KS-2',
    'US-GA-2',
    'US-WA-1',
    'US-TX-1',
    'CA-MTL-3',
    'EU-NL-1',
    'US-TX-4',
    'US-CA-2',
    'US-NC-1',
    'OC-AU-1',
    'US-DE-1',
    'EUR-IS-3',
    'CA-MTL-2',
    'AP-JP-1',
    'EUR-NO-1',
    'EU-FR-1',
    'US-KS-3',
    'US-GA-1',
] | str


_KNOWN_COUNTRIES: tuple[str, ...] = (
    "US", "CA", "DE", "FR", "NL", "SE", "CZ", "RO",
    "IS", "NO", "DK", "GB", "JP", "IN", "SG", "AU",
)


def _str_to_tuple[T: str](
    v: T | tuple[T, ...] | None,
    *,
    keep: tuple[T, ...] = (),
) -> T | tuple[T, ...] | None:
    match v:
        case str() as s if s in keep:
            return s
        case str() as s:
            return (s,)
        case _:
            return v


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
        >>> config = RunPod(data_center_ids="EU-RO-1")  # single datacenter

    Args:
        api_key: RunPod API key. Falls back to RUNPOD_API_KEY env var.
        cloud_type: Cloud type (SECURE or COMMUNITY). Default: SECURE.
        container_disk_gb: Container disk size in GB. Default: 50.
        volume_gb: Persistent volume size in GB. Default: 20.
        volume_mount_path: Volume mount path. Default: /workspace.
        data_center_ids: Preferred data center IDs — tuple of IDs, a single ID
            string, or ``"global"`` for auto-selection.
        ports: Port mappings (e.g., ["22/tcp", "8888/http"]). Default: ["22/tcp"].
        container_image: Override the container image for pods. When set, skips
            automatic image resolution from Docker Hub. Example:
            ``"runpod/base:1.0.3-cuda1210-ubuntu2204"``. Default: None (auto-select).
        registry_auth: Name of the container registry credential registered in RunPod
            account settings. Authenticates Docker Hub pulls to avoid rate limits.
            Set to None to skip. Default: "docker hub".
        min_inet_down: Minimum download speed in Mbps. None disables filter.
        min_inet_up: Minimum upload speed in Mbps. None disables filter.
        base_image: Docker Hub image family for automatic resolution. "base" uses
            ``runpod/base`` images (default), "pytorch" uses ``runpod/pytorch`` images
            which are typically pre-cached on RunPod hosts for faster startup.
            Ignored when ``container_image`` is set. Default: "base".
    """

    cluster_mode: ClusterMode = "individual"
    base_image: Literal["nvidia", "runpod-base", "runpod-pytorch"] = "runpod-base"
    global_networking: bool | None = None
    api_key: str | None = None
    cloud_type: Literal["community", "secure"] = "secure"
    ubuntu: Literal["20.04", "22.04", "24.04", "newest"] | str = "newest"
    container_disk_gb: int = 50
    volume_gb: int = 20
    volume_mount_path: str = "/workspace"
    data_center_ids: tuple[Datacenter, ...] | Datacenter | Literal["global"] = "global"
    country_codes: tuple[str, ...] | str | None = None
    exclude_country_codes: tuple[str, ...] | str = ()
    ports: tuple[str, ...] = ("22/tcp",)
    request_timeout: int = 30
    cpu_clock: Literal["3c", "5c"] | str = "3c"
    bid_multiplier: float = 1
    container_image: DockerImage | None = None
    registry_auth: str | None = "docker hub"
    min_inet_down: float | None = None
    min_inet_up: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_center_ids", _str_to_tuple(self.data_center_ids, keep=("global",)))
        object.__setattr__(self, "country_codes", _str_to_tuple(self.country_codes))
        object.__setattr__(self, "exclude_country_codes", _str_to_tuple(self.exclude_country_codes))

    def effective_country_codes(self) -> tuple[str, ...]:
        """Return countries to consider after applying exclusions.

        Empty tuple means "no constraint" (let RunPod choose).
        """
        excluded = frozenset(self.exclude_country_codes)
        match self.country_codes:
            case None if not excluded:
                return ()
            case None:
                return tuple(c for c in _KNOWN_COUNTRIES if c not in excluded)
            case tuple() as cs:
                return tuple(c for c in cs if c not in excluded)
            case _:
                return ()

    async def create_provider(self) -> RunPodProvider:
        from skyward.providers.runpod.provider import RunPodProvider
        return await RunPodProvider.create(self)

    @property
    def type(self) -> str: return "runpod"

    def default_options(self) -> None:
        return None

    @property
    def region(self) -> str:
        if self.data_center_ids == "global":
            return "global"
        return self.data_center_ids[0]
