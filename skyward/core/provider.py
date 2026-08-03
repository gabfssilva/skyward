"""Where a provider's blanks get filled in.

The providers themselves are :mod:`skyward.shared.providers` — plain values, so
that the daemon can rebuild one from a row without inheriting this module's
habits. What lives here is the one habit the daemon must not have: reading the
environment. An adapter is handed its credentials and has no other way to get
them, so somebody in the user's own process has to look them up, and that
somebody should be code the user can read rather than a hidden chain inside a
server they may not even be running.

Passing a credential explicitly always wins over the environment.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from configparser import ConfigParser
from pathlib import Path
from typing import Any

from msgspec import structs

from skyward.shared.providers import (
    AWS,
    GCP,
    Container,
    Hyperstack,
    JarvisLabs,
    Lambda,
    MassedCompute,
    Novita,
    Provider,
    RunPod,
    Salad,
    Scaleway,
    TensorDock,
    VastAI,
    Verda,
    Vultr,
    split,
    variables,
)


def resolve(provider: Provider) -> tuple[dict[str, str], dict[str, Any]]:
    """What the provider row is made of, once the environment has answered.

    Returns
    -------
    tuple[dict[str, str], dict[str, Any]]
        The credentials and the settings, as :func:`skyward.shared.providers.split`
        leaves them.
    """
    unset = frozenset(field.name for field in structs.fields(provider) if getattr(provider, field.name) is None)
    environment = {name: value for name, variable in variables(type(provider)).items() if name in unset and (value := os.environ.get(variable))}
    fallback = FALLBACKS.get(provider.kind) if unset - environment.keys() else None
    found = {**{name: value for name, value in fallback().items() if name in unset}, **environment} if fallback else environment
    return split(structs.replace(provider, **found) if found else provider)


def _aws_shared() -> Mapping[str, str]:
    """The static keys in ``~/.aws/credentials`` for ``AWS_PROFILE`` (default ``default``).

    Only static keys resolve here. SSO, assume-role, and process credentials need
    botocore's full chain, which the SDK deliberately does not depend on.
    """
    path = Path(os.environ.get("AWS_SHARED_CREDENTIALS_FILE") or Path.home() / ".aws" / "credentials")
    if not path.is_file():
        return {}
    parser = ConfigParser()
    parser.read(path)
    profile = os.environ.get("AWS_PROFILE", "default")
    section = dict(parser[profile]) if parser.has_section(profile) else {}
    return {field: section[f"aws_{field}"] for field in ("access_key_id", "secret_access_key", "session_token") if f"aws_{field}" in section}


def _gcp_service_account() -> Mapping[str, str]:
    """``GOOGLE_APPLICATION_CREDENTIALS`` names a file; its contents are what travel."""
    path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    return {"service_account_json": Path(path).read_text()} if path and Path(path).is_file() else {}


FALLBACKS: Mapping[str, Callable[[], Mapping[str, str]]] = {
    "aws": _aws_shared,
    "gcp": _gcp_service_account,
}
"""Providers whose credentials live somewhere an environment variable cannot name."""


__all__ = [
    "AWS",
    "GCP",
    "Container",
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
    "resolve",
]
