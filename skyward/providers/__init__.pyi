"""Cloud provider configurations."""

from typing import Any, Literal

class AWS:
    """Amazon Web Services cloud provider.

    Parameters
    ----------
    region
        AWS region. Default ``"us-east-1"``.
    ami
        Custom AMI ID. ``None`` uses Skyward's default Ubuntu image.
    ubuntu_version
        Ubuntu version for the default AMI.
    subnet_id
        VPC subnet ID. ``None`` uses default VPC.
    security_group_id
        Security group ID. ``None`` creates one automatically.
    instance_profile_arn
        IAM instance profile ARN. ``"auto"`` creates one automatically.
    username
        SSH username override.
    instance_timeout
        Seconds to wait for instance to become running.
    request_timeout
        HTTP request timeout in seconds.
    allocation_strategy
        Fleet allocation strategy.
    exclude_burstable
        Exclude burstable instance types (T-series).

    Examples
    --------
    >>> with sky.ComputePool(provider=sky.AWS(), accelerator="A100") as compute:
    ...     result = train(data) >> compute

    >>> sky.AWS(region="eu-west-1", exclude_burstable=True)
    """

    def __init__(
        self,
        *,
        region: str = "us-east-1",
        ami: str | None = None,
        ubuntu_version: Literal["20.04", "22.04", "24.04"] | str = "24.04",
        subnet_id: str | None = None,
        security_group_id: str | None = None,
        instance_profile_arn: str | Literal["auto"] | None = None,
        username: str | None = None,
        instance_timeout: int = 300,
        request_timeout: int = 30,
        allocation_strategy: Literal[
            "price-capacity-optimized", "capacity-optimized", "lowest-price"
        ] = "price-capacity-optimized",
        exclude_burstable: bool = False,
    ) -> None: ...
    @property
    def type(self) -> str: ...
    async def create_provider(self) -> Any: ...

class GCP:
    """Google Cloud Platform provider.

    Parameters
    ----------
    project
        GCP project ID. ``None`` uses Application Default Credentials.
    zone
        Compute Engine zone. Default ``"us-central1-a"``.
    network
        VPC network name. Default ``"default"``.
    subnet
        VPC subnet name. ``None`` uses default.
    disk_size_gb
        Boot disk size in GB.
    disk_type
        Boot disk type (``"pd-balanced"``, ``"pd-ssd"``, ``"pd-standard"``).
    instance_timeout
        Seconds to wait for instance to become running.
    service_account
        GCE service account email. ``None`` uses default.
    thread_pool_size
        Max concurrent API requests.

    Examples
    --------
    >>> sky.GCP(project="my-project", zone="us-west1-b")
    """

    def __init__(
        self,
        *,
        project: str | None = None,
        zone: str = "us-central1-a",
        network: str = "default",
        subnet: str | None = None,
        disk_size_gb: int = 200,
        disk_type: str = "pd-balanced",
        instance_timeout: int = 300,
        service_account: str | None = None,
        thread_pool_size: int = 8,
    ) -> None: ...
    @property
    def type(self) -> str: ...
    async def create_provider(self) -> Any: ...

class Hyperstack:
    """Hyperstack GPU cloud provider.

    Parameters
    ----------
    api_key
        Hyperstack API key. ``None`` reads from ``HYPERSTACK_API_KEY`` env var.
    region
        Region or tuple of regions to search. ``None`` searches all.
    image
        Custom OS image name. ``None`` uses Skyward's default.
    network_optimised
        Request network-optimised instances.
    instance_timeout
        Seconds to wait for instance to become running.
    request_timeout
        HTTP request timeout in seconds.
    teardown_timeout
        Max seconds to wait during teardown.

    Examples
    --------
    >>> sky.Hyperstack(region="CANADA-1")
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        region: str | tuple[str, ...] | None = None,
        image: str | None = None,
        network_optimised: bool = False,
        network_optimised_regions: tuple[str, ...] = ("CANADA-1", "US-1"),
        object_storage_region: str = "CANADA-1",
        object_storage_endpoint: str = "https://ca1.obj.nexgencloud.io",
        instance_timeout: int = 300,
        request_timeout: int = 30,
        teardown_timeout: int = 120,
        teardown_poll_interval: float = 2.0,
    ) -> None: ...
    @property
    def type(self) -> str: ...
    async def create_provider(self) -> Any: ...

class JarvisLabs:
    """Jarvis Labs GPU cloud provider.

    Uses the ``jarvislabs`` Python SDK (sync calls dispatched via
    ``ThreadPoolExecutor``). Supports IN1, IN2, and EU1 regions
    with per-minute billing. SSH keys auto-registered.

    Parameters
    ----------
    api_key
        API token. ``None`` reads from ``JL_API_KEY`` env var.
    region
        Preferred region (``"IN1"``, ``"IN2"``, ``"EU1"``).
        ``None`` auto-selects from GPU availability.
    template
        Framework template: ``"pytorch"``, ``"tensorflow"``,
        ``"jax"``, ``"vm"``.
    storage_gb
        Disk storage per instance in GB. EU1 and VM require
        minimum 100 GB.
    instance_timeout
        Auto-shutdown safety timeout in seconds.
    thread_pool_size
        Max worker threads for SDK calls.

    Examples
    --------
    >>> sky.JarvisLabs(region="IN2")
    >>> sky.JarvisLabs(template="vm", storage_gb=100)
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        region: str | None = None,
        template: str = "pytorch",
        storage_gb: int = 50,
        instance_timeout: int = 300,
        thread_pool_size: int = 8,
    ) -> None: ...
    @property
    def type(self) -> str: ...
    async def create_provider(self) -> Any: ...

class RunPod:
    """RunPod serverless GPU provider.

    Parameters
    ----------
    cluster_mode
        ``"instant"`` for RunPod's managed cluster, ``"individual"`` for
        individual pod provisioning.
    global_networking
        Enable global networking. ``None`` auto-detects.
    api_key
        RunPod API key. ``None`` reads from ``RUNPOD_API_KEY`` env var.
    cloud_type
        ``"secure"`` for verified data centers, ``"community"`` for
        community-hosted GPUs.
    ubuntu
        Ubuntu version for the base image.
    container_disk_gb
        Container root disk size in GB.
    volume_gb
        Persistent volume size in GB.
    volume_mount_path
        Mount path for the persistent volume.
    data_center_ids
        Specific data center IDs or ``"global"`` for any.
    ports
        Network ports to expose.
    provision_timeout
        Max seconds to wait for pod provisioning.
    bootstrap_timeout
        Max seconds to wait for bootstrap completion.
    bid_multiplier
        Multiplier for spot bid price.
    container_image
        Override the container image for pods. Skips automatic image
        resolution from Docker Hub when set.
    min_inet_down
        Minimum download speed in Mbps. ``None`` disables filter.
    min_inet_up
        Minimum upload speed in Mbps. ``None`` disables filter.

    Examples
    --------
    >>> sky.RunPod(cloud_type="secure", container_disk_gb=100)
    """

    def __init__(
        self,
        *,
        cluster_mode: Literal["instant", "individual"] = "individual",
        global_networking: bool | None = None,
        api_key: str | None = None,
        cloud_type: Literal["community", "secure"] = "secure",
        ubuntu: Literal["20.04", "22.04", "24.04", "newest"] | str = "newest",
        container_disk_gb: int = 50,
        volume_gb: int = 20,
        volume_mount_path: str = "/workspace",
        data_center_ids: tuple[str, ...] | Literal["global"] = "global",
        ports: tuple[str, ...] = ("22/tcp",),
        provision_timeout: float = 300.0,
        bootstrap_timeout: float = 600.0,
        instance_timeout: int = 300,
        request_timeout: int = 30,
        cpu_clock: Literal["3c", "5c"] | str = "3c",
        bid_multiplier: float = 1,
        container_image: str | None = None,
        registry_auth: str | None = "docker hub",
        min_inet_down: float | None = None,
        min_inet_up: float | None = None,
    ) -> None: ...
    @property
    def type(self) -> str: ...
    async def create_provider(self) -> Any: ...

class TensorDock:
    """TensorDock GPU marketplace provider.

    Parameters
    ----------
    api_key
        TensorDock API key. ``None`` reads from ``TENSORDOCK_API_KEY``.
    api_token
        TensorDock API token. ``None`` reads from ``TENSORDOCK_API_TOKEN``.
    location
        Preferred data center location.
    tier
        Data center tier (0-4). Higher tiers have better reliability.
    storage_gb
        Boot disk size in GB.
    operating_system
        OS image identifier.
    instance_timeout
        Seconds to wait for instance to become running.
    request_timeout
        HTTP request timeout in seconds.
    min_ram_gb
        Minimum system RAM in GB.
    min_vcpus
        Minimum vCPU count.

    Examples
    --------
    >>> sky.TensorDock(storage_gb=200, tier=2)
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_token: str | None = None,
        location: str | None = None,
        tier: Literal[0, 1, 2, 3, 4] | None = None,
        storage_gb: int = 100,
        operating_system: str = "ubuntu2404",
        instance_timeout: int = 300,
        request_timeout: int = 120,
        min_ram_gb: int | None = None,
        min_vcpus: int | None = None,
    ) -> None: ...
    @property
    def type(self) -> str: ...
    async def create_provider(self) -> Any: ...

class VastAI:
    """Vast.ai GPU marketplace provider.

    Parameters
    ----------
    api_key
        Vast.ai API key. ``None`` reads from ``VASTAI_API_KEY`` env var.
    min_reliability
        Minimum machine reliability score (0.0-1.0).
    verified_only
        Only use verified machines.
    min_cuda
        Minimum CUDA version.
    geolocation
        Geographic filter (e.g., ``"US"``, ``"EU"``).
    bid_multiplier
        Multiplier for spot bid price above minimum.
    docker_image
        Custom Docker image. ``None`` uses Skyward's default.
    disk_gb
        Disk space in GB.
    use_overlay
        Enable overlay networking for multi-node clusters.
    overlay_timeout
        Seconds to wait for overlay network setup.
    require_direct_port
        Require direct SSH port (no NAT).
    min_inet_down
        Minimum download speed in Mbps. ``None`` disables filter.
    min_inet_up
        Minimum upload speed in Mbps. ``None`` disables filter.

    Examples
    --------
    >>> sky.VastAI(min_reliability=0.98, verified_only=True)
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        min_reliability: float = 0.95,
        verified_only: bool = True,
        min_cuda: float = 12.0,
        geolocation: str | None = None,
        bid_multiplier: float = 1.2,
        instance_timeout: int = 300,
        request_timeout: int = 30,
        docker_image: str | None = None,
        disk_gb: int = 100,
        use_overlay: bool = True,
        overlay_timeout: int = 120,
        require_direct_port: bool = False,
        min_inet_down: float | None = None,
        min_inet_up: float | None = None,
    ) -> None: ...
    @property
    def type(self) -> str: ...
    async def create_provider(self) -> Any: ...
    @classmethod
    def ubuntu(
        cls,
        version: Literal["22.04", "24.04", "26.04"] | str = "24.04",
        cuda: Literal["12.9.1", "13.1.0", "13.0.1"] | str = "12.9.1",
        cuda_dist: Literal["devel", "runtime"] = "runtime",
    ) -> str:
        """Generate an NVIDIA CUDA Docker image name.

        Parameters
        ----------
        version
            Ubuntu version.
        cuda
            CUDA toolkit version.
        cuda_dist
            CUDA distribution type.

        Returns
        -------
        str
            Docker image name (e.g., ``"nvidia/cuda:12.9.1-runtime-ubuntu24.04"``).
        """
        ...

class Verda:
    """Verda GPU cloud provider.

    Parameters
    ----------
    region
        Verda region. Default ``"FIN-01"``.
    client_id
        OAuth client ID. ``None`` reads from ``VERDA_CLIENT_ID``.
    client_secret
        OAuth client secret. ``None`` reads from ``VERDA_CLIENT_SECRET``.
    ssh_key_id
        Pre-registered SSH key ID.
    instance_timeout
        Seconds to wait for instance to become running.
    request_timeout
        HTTP request timeout in seconds.

    Examples
    --------
    >>> sky.Verda(region="FIN-01")
    """

    def __init__(
        self,
        *,
        region: str = "FIN-01",
        client_id: str | None = None,
        client_secret: str | None = None,
        ssh_key_id: str | None = None,
        instance_timeout: int = 300,
        request_timeout: int = 30,
    ) -> None: ...
    @property
    def type(self) -> str: ...
    async def create_provider(self) -> Any: ...

class Scaleway:
    """Scaleway GPU cloud provider.

    Scaleway provides GPU instances (L4, L40S, H100, B300) in European
    zones. Auth uses a secret key passed via ``X-Auth-Token`` header.

    Parameters
    ----------
    secret_key
        Scaleway secret key (UUID). ``None`` reads from ``SCW_SECRET_KEY`` env var.
    project_id
        Scaleway project ID (UUID). ``None`` reads from ``SCW_DEFAULT_PROJECT_ID`` env var.
    zone
        Availability zone. ``None`` searches all GPU zones automatically.
    image
        OS image UUID override. ``None`` auto-selects Ubuntu GPU image.
    instance_timeout
        Auto-shutdown safety timeout in seconds.
    request_timeout
        HTTP request timeout in seconds.

    Examples
    --------
    >>> sky.Scaleway(zone="fr-par-2")
    """

    def __init__(
        self,
        *,
        secret_key: str | None = None,
        project_id: str | None = None,
        zone: str | None = None,
        image: str | None = None,
        instance_timeout: int = 300,
        request_timeout: int = 30,
    ) -> None: ...
    @property
    def type(self) -> str: ...
    async def create_provider(self) -> Any: ...

class Vultr:
    """Vultr GPU cloud provider.

    Supports two modes: Cloud GPU (virtual instances with ``vcg-*`` plans)
    and Bare Metal (dedicated servers with ``vbm-*`` plans).

    Parameters
    ----------
    api_key
        API key. ``None`` reads from ``VULTR_API_KEY`` env var.
    mode
        ``"cloud"`` for virtual GPU instances (default),
        ``"bare-metal"`` for dedicated physical servers.
    region
        Vultr region ID (e.g., ``"ewr"``, ``"ord"``).
    os_id
        OS image ID. Default ``2284`` (Ubuntu 24.04).
    instance_timeout
        Safety timeout in seconds.
    request_timeout
        HTTP request timeout in seconds.

    Examples
    --------
    >>> sky.Vultr(region="ewr")
    >>> sky.Vultr(mode="bare-metal", region="ord")
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        mode: Literal["cloud", "bare-metal"] = "cloud",
        region: str = "ewr",
        os_id: int = 2284,
        instance_timeout: int = 300,
        request_timeout: int = 30,
    ) -> None: ...
    @property
    def type(self) -> str: ...
    async def create_provider(self) -> Any: ...

class Container:
    """Local/remote container provider for development and testing.

    Run Skyward pools using Docker or Podman containers instead of
    cloud instances. Useful for local development and CI.

    Parameters
    ----------
    image
        Container image. Supports ``{python_version}`` placeholder.
    ssh_user
        SSH user inside the container.
    binary
        Container runtime binary (``"docker"`` or ``"podman"``).
    container_prefix
        Prefix for container names. ``None`` uses auto-generated names.
    network
        Docker network name. ``None`` uses default.

    Examples
    --------
    >>> with sky.ComputePool(provider=sky.Container(), nodes=2) as compute:
    ...     result = train(data) >> compute
    """

    def __init__(
        self,
        *,
        image: str = ...,
        ssh_user: str = "root",
        binary: str = "docker",
        container_prefix: str | None = None,
        network: str | None = None,
    ) -> None: ...
    @property
    def type(self) -> str: ...
    async def create_provider(self) -> Any: ...

__all__ = [
    "AWS",
    "GCP",
    "Hyperstack",
    "JarvisLabs",
    "RunPod",
    "Scaleway",
    "TensorDock",
    "VastAI",
    "Verda",
    "Vultr",
    "Container",
]
