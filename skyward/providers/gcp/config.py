"""GCP provider configuration.

Immutable configuration dataclass for GCP Compute Engine provider.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass

from skyward.api.provider import ProviderConfig

if typing.TYPE_CHECKING:
    from skyward.providers.gcp.provider import GCPProvider


@dataclass(frozen=True, slots=True)
class GCP(ProviderConfig):
    """GCP Compute Engine provider configuration.

    Immutable configuration that defines how to connect to GCP and
    provision Compute Engine resources. All fields have sensible defaults.

    SSH keys are injected into instance metadata during provisioning.
    The project is auto-detected from Application Default Credentials
    or the GOOGLE_CLOUD_PROJECT environment variable if not specified.

    Example:
        >>> from skyward.providers.gcp import GCP
        >>> config = GCP(zone="us-central1-a")

    Args:
        project: GCP project ID. Auto-detected from ADC or GOOGLE_CLOUD_PROJECT.
        zone: Compute Engine zone for resources. Default: us-central1-a.
        network: VPC network name. Default: default.
        subnet: Specific subnet. If None, uses auto-mode subnet for the region.
        disk_size_gb: Boot disk size in GB. Default: 200.
        disk_type: Boot disk type. Default: pd-balanced.
        instance_timeout: Safety timeout in seconds. Default: 300.
        service_account: GCE service account email. If None, uses default.
    """

    project: str | None = None
    zone: str = "us-central1-a"
    network: str = "default"
    subnet: str | None = None
    disk_size_gb: int = 200
    disk_type: str = "pd-balanced"
    instance_timeout: int = 300
    service_account: str | None = None
    thread_pool_size: int = 8

    @property
    def type(self) -> str: return "gcp"

    async def create_provider(self) -> GCPProvider:
        from skyward.providers.gcp.provider import GCPProvider
        return await GCPProvider.create(self)
