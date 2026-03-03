"""Factory functions for common S3-compatible storage providers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from skyward.storage import Credential, Storage


def R2(*, account_id: str, access_key: Credential, secret_key: Credential) -> Storage:
    """Cloudflare R2 storage.

    Parameters
    ----------
    account_id
        Cloudflare account ID.
    access_key
        R2 access key ID.
    secret_key
        R2 secret access key.
    """
    from skyward.storage import Storage

    return Storage(
        endpoint=f"https://{account_id}.r2.cloudflarestorage.com",
        access_key=access_key,
        secret_key=secret_key,
    )


def S3(
    *,
    region: str = "us-east-1",
    access_key: Credential | None = None,
    secret_key: Credential | None = None,
) -> Storage:
    """Amazon S3 storage.

    Parameters
    ----------
    region
        AWS region.
    access_key
        AWS access key ID. ``None`` defers to environment / IAM credentials.
    secret_key
        AWS secret access key. ``None`` defers to environment / IAM credentials.
    """
    from skyward.storage import Storage

    return Storage(
        endpoint=f"https://s3.{region}.amazonaws.com",
        access_key=access_key,
        secret_key=secret_key,
    )


def GCS(*, access_key: Credential, secret_key: Credential) -> Storage:
    """Google Cloud Storage (S3-compatible interop).

    Parameters
    ----------
    access_key
        HMAC access key.
    secret_key
        HMAC secret key.
    """
    from skyward.storage import Storage

    return Storage(
        endpoint="https://storage.googleapis.com",
        access_key=access_key,
        secret_key=secret_key,
    )


def Wasabi(*, region: str = "us-east-1", access_key: Credential, secret_key: Credential) -> Storage:
    """Wasabi hot cloud storage.

    Parameters
    ----------
    region
        Wasabi region.
    access_key
        Wasabi access key.
    secret_key
        Wasabi secret key.
    """
    from skyward.storage import Storage

    return Storage(
        endpoint=f"https://s3.{region}.wasabisys.com",
        access_key=access_key,
        secret_key=secret_key,
    )


def Backblaze(*, region: str, key_id: Credential, app_key: Credential) -> Storage:
    """Backblaze B2 storage (S3-compatible).

    Parameters
    ----------
    region
        B2 region (e.g. ``us-west-004``).
    key_id
        B2 application key ID.
    app_key
        B2 application key.
    """
    from skyward.storage import Storage

    return Storage(
        endpoint=f"https://s3.{region}.backblazeb2.com",
        access_key=key_id,
        secret_key=app_key,
    )


def Hyperstack(
    *,
    api_key: str | None = None,
    region: str = "CANADA-1",
    endpoint: str | None = None,
) -> Storage:
    """Hyperstack object storage with auto-provisioned credentials.

    Creates an ephemeral access key via the Hyperstack API on first use
    and deletes it when the context manager exits.

    Parameters
    ----------
    api_key
        Hyperstack API key. Falls back to ``HYPERSTACK_API_KEY`` env var.
    region
        Object storage region (e.g. ``CANADA-1``).
    endpoint
        S3 endpoint override. Defaults to the region's standard endpoint.
    """
    from skyward.providers.hyperstack.config import Hyperstack as HyperstackConfig
    from skyward.storage import _ON_CLOSE, Storage

    config = HyperstackConfig(api_key=api_key, object_storage_region=region)
    resolved_endpoint = endpoint or config.object_storage_endpoint

    _created: dict[str, Any] = {}

    async def _create_and_get_access_key() -> str:
        from skyward.providers.hyperstack.client import HyperstackClient, get_api_key

        resolved_api_key = get_api_key(config)
        async with HyperstackClient(resolved_api_key, config=config) as client:
            result = await client.create_access_key(region=config.object_storage_region)
        _created.update({
            "access_key": result["access_key"],
            "secret_key": result.get("secret_key", ""),
            "id": result["id"],
            "api_key": resolved_api_key,
        })
        return _created["access_key"]

    async def _get_secret_key() -> str:
        return _created["secret_key"]

    async def _delete_key() -> None:
        if "id" not in _created:
            return
        from skyward.providers.hyperstack.client import HyperstackClient

        async with HyperstackClient(_created["api_key"], config=config) as client:
            await client.delete_access_key(_created["id"])

    storage = Storage(
        endpoint=resolved_endpoint,
        access_key=_create_and_get_access_key,
        secret_key=_get_secret_key,
        path_style=True,
    )
    _ON_CLOSE[id(storage)] = [_delete_key]
    return storage
