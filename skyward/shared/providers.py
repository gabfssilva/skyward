"""Who to rent from, and what it takes to log in.

One struct per provider, and it is the same object on both sides: the user
constructs it, the client splits it into the secrets and the settings a provider
row is made of, and the daemon converts the settings back into it before handing
it to the adapter. Field names and defaults therefore exist once — an adapter
reading ``config.request_timeout`` is reading the very field the user saw.

Nothing here reads the environment or imports a provider sdk: a client talking to
a remote daemon must be able to write ``sky.AWS()`` with neither. Resolving what
was left unset is :mod:`skyward.core.provider`'s job, in the user's own process.

The neighbouring :mod:`skyward.shared.provider` is the other half of the word: it
is what an adapter must implement, while this is what a user writes.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Annotated, Any, ClassVar, Literal, get_args, get_type_hints

from msgspec import Struct, structs, to_builtins

SCALEWAY_ZONES = (
    "fr-par-1",
    "fr-par-2",
    "fr-par-3",
    "nl-ams-1",
    "nl-ams-2",
    "nl-ams-3",
    "pl-waw-1",
    "pl-waw-2",
    "pl-waw-3",
)


class Credential:
    """Marks a field as a secret.

    Credentials travel to the daemon apart from the rest and are stored on the
    provider row rather than on the compute — the API serves a provider's config
    back, and never its credentials.
    """


@dataclass(frozen=True, slots=True)
class Env:
    """The environment variable that answers for a field the user left unset."""

    variable: str


class Provider(Struct, frozen=True, kw_only=True):
    """A provider account: which cloud, and what it takes to log in.

    Attributes
    ----------
    name : str
        The account's alias, and the identity of the provider row. Two accounts
        of the same kind coexist by having two names. Empty is the kind itself.
    """

    kind: ClassVar[str]

    name: str = ""

    def __post_init__(self) -> None:
        for f in structs.fields(self):
            value = getattr(self, f.name)
            if isinstance(value, Sequence) and not isinstance(value, (str, tuple)):
                structs.force_setattr(self, f.name, tuple(value))


@cache
def secrets(kind: type[Provider]) -> tuple[str, ...]:
    """The fields of ``kind`` marked :class:`Credential`."""
    return tuple(name for name, hint in _hints(kind).items() if Credential in get_args(hint))


@cache
def variables(kind: type[Provider]) -> Mapping[str, str]:
    """Which environment variable answers for which field, for the fields that name one."""
    return {name: marker.variable for name, hint in _hints(kind).items() for marker in get_args(hint) if isinstance(marker, Env)}


def split(provider: Provider) -> tuple[dict[str, str], dict[str, Any]]:
    """Take a provider apart into what a provider row is made of.

    Returns
    -------
    tuple[dict[str, str], dict[str, Any]]
        The credentials that were set, and every other field. The alias is
        neither: it names the row rather than living in it.
    """
    keys = secrets(type(provider))
    credentials = {key: value for key in keys if (value := getattr(provider, key)) is not None}
    config = {field.name: to_builtins(getattr(provider, field.name)) for field in structs.fields(provider) if field.name not in keys and field.name != "name"}
    return credentials, config


def _hints(kind: type[Provider]) -> Mapping[str, Any]:
    names = {field.name for field in structs.fields(kind)}
    return {name: hint for name, hint in get_type_hints(kind, include_extras=True).items() if name in names}


def _many(value: str | Sequence[str] | None, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    match value:
        case None:
            return default
        case str() as one:
            return (one,)
        case many:
            return tuple(many)


class AWS(Provider, frozen=True, kw_only=True):
    """Amazon EC2, through the fleet API.

    Only static keys resolve from ``~/.aws/credentials``. SSO, assume-role and
    process credentials need botocore's full chain, which the SDK deliberately
    does not depend on.
    """

    kind: ClassVar[str] = "aws"

    access_key_id: Annotated[str | None, Credential, Env("AWS_ACCESS_KEY_ID")] = None
    secret_access_key: Annotated[str | None, Credential, Env("AWS_SECRET_ACCESS_KEY")] = None
    session_token: Annotated[str | None, Credential, Env("AWS_SESSION_TOKEN")] = None
    region: str | Sequence[str] = "us-east-1"
    ami: str | None = None
    ubuntu_version: str = "24.04"
    subnet_id: str | None = None
    security_group_id: str | None = None
    instance_profile_arn: str | None = None
    username: str | None = None
    disk_gb: int = 100
    instance_timeout: int = 300
    request_timeout: int = 30
    allocation_strategy: Literal["price-capacity-optimized", "capacity-optimized", "lowest-price"] = "price-capacity-optimized"
    exclude_burstable: bool = False

    @property
    def regions(self) -> tuple[str, ...]:
        return _many(self.region)


class GCP(Provider, frozen=True, kw_only=True):
    """Google Compute Engine.

    ``GOOGLE_APPLICATION_CREDENTIALS`` names a file; its contents are what travel.
    """

    kind: ClassVar[str] = "gcp"

    service_account_json: Annotated[str | None, Credential] = None
    project: Annotated[str | None, Env("GOOGLE_CLOUD_PROJECT")] = None
    zone: str | Sequence[str] = "us-central1-a"
    network: str = "default"
    subnet: str | None = None
    disk_size_gb: int = 200
    disk_type: Literal["pd-balanced", "pd-ssd", "pd-standard", "pd-extreme", "hyperdisk-balanced"] = "pd-balanced"
    instance_timeout: int = 300
    service_account: str | None = None
    thread_pool_size: int = 8

    @property
    def zones(self) -> tuple[str, ...]:
        return _many(self.zone)


class Hyperstack(Provider, frozen=True, kw_only=True):
    """NexGen Cloud's Hyperstack."""

    kind: ClassVar[str] = "hyperstack"

    api_key: Annotated[str | None, Credential, Env("HYPERSTACK_API_KEY")] = None
    region: str | Sequence[str] | None = None
    image: str | None = None
    network_optimised: bool = False
    network_optimised_regions: Sequence[str] = ("CANADA-1", "US-1")
    object_storage_region: str = "CANADA-1"
    object_storage_endpoint: str = "https://ca1.obj.nexgencloud.io"
    instance_timeout: int = 300
    request_timeout: int = 30
    teardown_timeout: float = 120.0
    teardown_poll_interval: float = 2.0

    @property
    def regions(self) -> tuple[str, ...]:
        return _many(self.region)


class JarvisLabs(Provider, frozen=True, kw_only=True):
    """JarvisLabs."""

    kind: ClassVar[str] = "jarvislabs"

    api_key: Annotated[str | None, Credential, Env("JL_API_KEY")] = None
    region: str | None = None
    template: str = "pytorch"
    storage_gb: int = 50
    instance_timeout: int = 300
    thread_pool_size: int = 8


class Lambda(Provider, frozen=True, kw_only=True):
    """Lambda Cloud."""

    kind: ClassVar[str] = "lambda"

    api_key: Annotated[str | None, Credential, Env("LAMBDA_API_KEY")] = None
    region: str | None = None
    request_timeout: int = 30


class Salad(Provider, frozen=True, kw_only=True):
    """Salad, reached through its own container gateway.

    ``organization`` and ``project`` are not optional to the provider — they are
    the account's namespace — only optional here, where the environment may
    answer for them.
    """

    kind: ClassVar[str] = "salad"

    api_key: Annotated[str | None, Credential, Env("SALAD_API_KEY")] = None
    organization: Annotated[str | None, Env("SALAD_ORGANIZATION")] = None
    project: Annotated[str | None, Env("SALAD_PROJECT")] = None
    priority: Literal["high", "medium", "low", "batch"] = "low"
    country_codes: str | Sequence[str] | None = None
    image: str | None = None
    storage_gb: int = 50
    request_timeout: int = 30
    allocation_timeout: float = 300.0
    poll_interval: float = 2.0

    @property
    def countries(self) -> tuple[str, ...]:
        return _many(self.country_codes)


class MassedCompute(Provider, frozen=True, kw_only=True):
    """Massed Compute."""

    kind: ClassVar[str] = "massed_compute"

    api_key: Annotated[str | None, Credential, Env("MASSED_API_KEY")] = None
    image_id: int = 184
    request_timeout: int = 30


class Novita(Provider, frozen=True, kw_only=True):
    """Novita AI."""

    kind: ClassVar[str] = "novita"

    api_key: Annotated[str | None, Credential, Env("NOVITA_API_KEY")] = None
    cluster_id: str | None = None
    rootfs_size: int = 50
    docker_image: str | None = None
    min_cuda_version: str | None = None
    request_timeout: int = 30


class RunPod(Provider, frozen=True, kw_only=True):
    """RunPod, on the REST v2 API.

    ``data_center_ids`` takes ``"global"`` as a mode rather than as a member: it
    is the whole fleet, not a data center named global.
    """

    kind: ClassVar[str] = "runpod"

    api_key: Annotated[str | None, Credential, Env("RUNPOD_API_KEY")] = None
    cluster_mode: Literal["individual", "cluster"] = "individual"
    cloud_type: Literal["secure", "community"] = "secure"
    base_image: Literal["nvidia", "runpod_base", "runpod_pytorch"] = "nvidia"
    container_image: str | None = None
    ubuntu: str = "newest"
    container_disk_gb: int = 50
    volume_gb: int = 20
    volume_mount_path: str = "/workspace"
    data_center_ids: str | Sequence[str] = "global"
    country_codes: str | Sequence[str] | None = None
    exclude_country_codes: str | Sequence[str] = ()
    ports: Sequence[str] = ("22/tcp",)
    bid_multiplier: float = 1.0
    registry_auth: str | None = "docker hub"
    min_inet_down: float | None = None
    min_inet_up: float | None = None
    global_networking: bool | None = None
    request_timeout: int = 30
    cpu_clock: str = "3c"

    @property
    def centers(self) -> tuple[str, ...]:
        return () if self.data_center_ids == "global" else _many(self.data_center_ids)

    @property
    def countries(self) -> tuple[str, ...]:
        return _many(self.country_codes)

    @property
    def excluded(self) -> frozenset[str]:
        return frozenset(_many(self.exclude_country_codes))


class Scaleway(Provider, frozen=True, kw_only=True):
    """Scaleway instances."""

    kind: ClassVar[str] = "scaleway"

    secret_key: Annotated[str | None, Credential, Env("SCW_SECRET_KEY")] = None
    project_id: Annotated[str | None, Credential, Env("SCW_DEFAULT_PROJECT_ID")] = None
    zone: str | Sequence[str] | None = None
    image: str | None = None
    instance_timeout: int = 300
    request_timeout: int = 30

    @property
    def zones(self) -> tuple[str, ...]:
        return _many(self.zone, SCALEWAY_ZONES)


class TensorDock(Provider, frozen=True, kw_only=True):
    """TensorDock."""

    kind: ClassVar[str] = "tensordock"

    api_key: Annotated[str | None, Credential, Env("TENSORDOCK_API_KEY")] = None
    api_token: Annotated[str | None, Credential, Env("TENSORDOCK_API_TOKEN")] = None
    location: str | None = None
    tier: int | None = None
    storage_gb: int = 100
    operating_system: str = "ubuntu2404"
    instance_timeout: int = 300
    request_timeout: int = 120
    min_ram_gb: int | None = None
    min_vcpus: int | None = None


class VastAI(Provider, frozen=True, kw_only=True):
    """Vast.ai's marketplace."""

    kind: ClassVar[str] = "vastai"

    api_key: Annotated[str | None, Credential, Env("VAST_API_KEY")] = None
    min_reliability: float = 0.95
    verified_only: bool = True
    min_cuda: float = 12.0
    geolocation: str | None = None
    bid_multiplier: float = 1.2
    instance_timeout: int = 300
    request_timeout: int = 30
    docker_image: str | None = None
    disk_gb: float = 100.0
    overlay_timeout: int = 120
    require_direct_port: bool = False
    min_inet_down: float | None = None
    min_inet_up: float | None = None
    limit: int = 500


class Verda(Provider, frozen=True, kw_only=True):
    """Verda."""

    kind: ClassVar[str] = "verda"

    client_id: Annotated[str | None, Credential, Env("VERDA_CLIENT_ID")] = None
    client_secret: Annotated[str | None, Credential, Env("VERDA_CLIENT_SECRET")] = None
    region: str = "FIN-01"
    ssh_key_id: str | None = None
    image: str | None = None
    cuda: str = "13.0"
    instance_timeout: int = 300
    request_timeout: int = 30


class Vultr(Provider, frozen=True, kw_only=True):
    """Vultr, cloud instances or bare metal."""

    kind: ClassVar[str] = "vultr"

    api_key: Annotated[str | None, Credential, Env("VULTR_API_KEY")] = None
    mode: Literal["cloud", "bare_metal"] = "cloud"
    region: str | None = None
    os_id: int = 2284
    instance_timeout: int = 300
    request_timeout: int = 30


class Container(Provider, frozen=True, kw_only=True):
    """Local containers. The one provider that needs no credentials."""

    kind: ClassVar[str] = "container"

    image: str = "ghcr.io/gabfssilva/skyward:py{python_version}"
    ssh_user: str = "root"
    binary: str = "docker"
    container_prefix: str | None = None
    network: str | None = None


class Fake(Provider, frozen=True, kw_only=True):
    """A catalog with no cloud behind it, for tests."""

    kind: ClassVar[str] = "fake"

    region: str = "fake-1"


__all__ = [
    "AWS",
    "GCP",
    "SCALEWAY_ZONES",
    "Container",
    "Credential",
    "Env",
    "Fake",
    "Hyperstack",
    "JarvisLabs",
    "Lambda",
    "MassedCompute",
    "Novita",
    "Provider",
    "RunPod",
    "Salad",
    "Scaleway",
    "TensorDock",
    "VastAI",
    "Verda",
    "Vultr",
    "secrets",
    "split",
    "variables",
]
