"""Factory functions for common S3-compatible storage providers."""

from __future__ import annotations


def R2(*, account_id: str, access_key: str, secret_key: str) -> Storage:
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


def S3(*, region: str = "us-east-1", access_key: str | None = None, secret_key: str | None = None) -> Storage:
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


def GCS(*, access_key: str, secret_key: str) -> Storage:
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


def Wasabi(*, region: str = "us-east-1", access_key: str, secret_key: str) -> Storage:
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


def Backblaze(*, region: str, key_id: str, app_key: str) -> Storage:
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


if __name__ != "__main__":
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from skyward.storage import Storage
