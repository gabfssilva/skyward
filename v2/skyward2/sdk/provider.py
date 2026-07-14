"""Who to rent from, and what it takes to log in.

The factories below read the environment, and that is the whole reason they
exist: the daemon never does. An adapter is handed its credentials through
``create()`` and has no other way to get them, so somebody in the user's own
process has to look them up and send them — and that somebody should be code the
user can read, not a hidden chain inside a server they may not even be running.

Passing a credential explicitly always wins over the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Provider:
    """A provider kind, its credentials, and whatever configuration it reads."""

    kind: str
    credentials: dict[str, str] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            object.__setattr__(self, "name", self.kind)


def AWS(  # noqa: N802
    access_key_id: str | None = None,
    secret_access_key: str | None = None,
    session_token: str | None = None,
    regions: tuple[str, ...] | None = None,
    name: str = "",
) -> Provider:
    return _provider(
        "aws",
        name,
        _some(
            access_key_id=access_key_id or os.environ.get("AWS_ACCESS_KEY_ID"),
            secret_access_key=secret_access_key or os.environ.get("AWS_SECRET_ACCESS_KEY"),
            session_token=session_token or os.environ.get("AWS_SESSION_TOKEN"),
        ),
        _some(regions=regions),
    )


def GCP(  # noqa: N802
    service_account_json: str | None = None,
    project: str | None = None,
    zones: tuple[str, ...] | None = None,
    name: str = "",
) -> Provider:
    """``GOOGLE_APPLICATION_CREDENTIALS`` names a file; its contents are what travel."""
    credentials = service_account_json or _file(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
    return _provider(
        "gcp",
        name,
        _some(service_account_json=credentials),
        _some(project=project or os.environ.get("GOOGLE_CLOUD_PROJECT"), zones=zones),
    )


def Hyperstack(api_key: str | None = None, name: str = "") -> Provider:  # noqa: N802
    return _key("hyperstack", "HYPERSTACK_API_KEY", api_key, name)


def JarvisLabs(api_key: str | None = None, name: str = "") -> Provider:  # noqa: N802
    return _key("jarvislabs", "JL_API_KEY", api_key, name)


def Lambda(api_key: str | None = None, name: str = "") -> Provider:  # noqa: N802
    return _key("lambda", "LAMBDA_API_KEY", api_key, name)


def MassedCompute(api_key: str | None = None, name: str = "") -> Provider:  # noqa: N802
    return _key("massed_compute", "MASSED_API_KEY", api_key, name)


def Novita(api_key: str | None = None, cluster_id: str | None = None, name: str = "") -> Provider:  # noqa: N802
    return _provider(
        "novita",
        name,
        _some(api_key=api_key or os.environ.get("NOVITA_API_KEY")),
        _some(cluster_id=cluster_id),
    )


def RunPod(api_key: str | None = None, name: str = "") -> Provider:  # noqa: N802
    return _key("runpod", "RUNPOD_API_KEY", api_key, name)


def Scaleway(secret_key: str | None = None, zones: tuple[str, ...] | None = None, name: str = "") -> Provider:  # noqa: N802
    return _provider(
        "scaleway",
        name,
        _some(secret_key=secret_key or os.environ.get("SCW_SECRET_KEY")),
        _some(zones=zones),
    )


def TensorDock(api_token: str | None = None, storage_gb: int | None = None, name: str = "") -> Provider:  # noqa: N802
    return _provider(
        "tensordock",
        name,
        _some(api_token=api_token or os.environ.get("TENSORDOCK_API_TOKEN")),
        _some(storage_gb=storage_gb),
    )


def VastAI(  # noqa: N802
    api_key: str | None = None,
    verified_only: bool | None = None,
    min_reliability: float | None = None,
    limit: int | None = None,
    name: str = "",
) -> Provider:
    return _provider(
        "vastai",
        name,
        _some(api_key=api_key or os.environ.get("VAST_API_KEY")),
        _some(verified_only=verified_only, min_reliability=min_reliability, limit=limit),
    )


def Verda(client_id: str | None = None, client_secret: str | None = None, name: str = "") -> Provider:  # noqa: N802
    return _provider(
        "verda",
        name,
        _some(
            client_id=client_id or os.environ.get("VERDA_CLIENT_ID"),
            client_secret=client_secret or os.environ.get("VERDA_CLIENT_SECRET"),
        ),
        {},
    )


def Vultr(api_key: str | None = None, name: str = "") -> Provider:  # noqa: N802
    return _key("vultr", "VULTR_API_KEY", api_key, name)


def Container(binary: str | None = None, name: str = "") -> Provider:  # noqa: N802
    """Local containers. The one provider that needs no credentials."""
    return _provider("container", name, {}, _some(binary=binary))


def _key(kind: str, variable: str, api_key: str | None, name: str) -> Provider:
    return _provider(kind, name, _some(api_key=api_key or os.environ.get(variable)), {})


def _provider(kind: str, name: str, credentials: dict[str, str], config: dict[str, Any]) -> Provider:
    return Provider(kind=kind, credentials=credentials, config=config, name=name)


def _some[T](**fields: T | None) -> dict[str, T]:
    """Only what was actually given; the adapter's own defaults answer for the rest."""
    return {name: value for name, value in fields.items() if value is not None}


def _file(path: str | None) -> str | None:
    return Path(path).read_text() if path and Path(path).is_file() else None
