"""S3-based object store for AWS deployments."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client


class ObjectStore(Protocol):
    """Protocol para object store."""

    def put(self, key: str, data: bytes) -> None:
        """Store data."""
        ...

    def get(self, key: str) -> bytes:
        """Retrieve data."""
        ...

    def delete(self, key: str) -> None:
        """Delete data."""
        ...

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        ...


class S3ObjectStore:
    """Object store using S3 for data persistence."""

    def __init__(self, bucket: str, prefix: str = "objects/") -> None:
        """Initialize S3 object store.

        Args:
            bucket: S3 bucket name.
            prefix: Key prefix for all objects.
        """
        self.bucket = bucket
        self.prefix = prefix

    @cached_property
    def _s3(self) -> S3Client:
        """Cliente S3 com inicialização lazy."""
        import boto3

        return boto3.client("s3")

    def _full_key(self, key: str) -> str:
        """Get full S3 key with prefix."""
        return f"{self.prefix}{key}"

    def put(self, key: str, data: bytes) -> None:
        """Store data in S3.

        Args:
            key: Unique identifier for the object.
            data: Bytes to store.
        """
        self._s3.put_object(
            Bucket=self.bucket,
            Key=self._full_key(key),
            Body=data,
        )

    def get(self, key: str) -> bytes:
        """Retrieve data from S3.

        Args:
            key: Unique identifier for the object.

        Returns:
            The stored bytes.

        Raises:
            KeyError: If the key does not exist.
        """
        try:
            response = self._s3.get_object(
                Bucket=self.bucket,
                Key=self._full_key(key),
            )
            return response["Body"].read()
        except self._s3.exceptions.NoSuchKey as e:
            raise KeyError(f"Object not found: {key}") from e

    def delete(self, key: str) -> None:
        """Delete object from S3.

        Args:
            key: Unique identifier for the object.
        """
        self._s3.delete_object(
            Bucket=self.bucket,
            Key=self._full_key(key),
        )

    def exists(self, key: str) -> bool:
        """Check if object exists in S3.

        Args:
            key: Unique identifier for the object.

        Returns:
            True if the key exists, False otherwise.
        """
        try:
            self._s3.head_object(
                Bucket=self.bucket,
                Key=self._full_key(key),
            )
            return True
        except Exception:
            return False
