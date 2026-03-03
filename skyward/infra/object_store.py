"""S3-compatible object store backed by aioboto3."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class S3ObjectStore:
    """ObjectStore implementation for S3-compatible backends.

    Wraps an aioboto3 S3 client and implements the ObjectStore protocol
    defined in ``skyward.providers.provider``.
    """

    _s3: Any

    async def upload_file(self, bucket: str, key: str, path: Path) -> None:
        content = path.read_bytes()
        await self._s3.put_object(Bucket=bucket, Key=key, Body=content)

    async def download_file(self, bucket: str, key: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        await self._s3.download_file(bucket, key, str(path))

    async def list_objects(self, bucket: str, prefix: str) -> list[str]:
        keys: list[str] = []
        continuation_token: str | None = None

        while True:
            kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
            if continuation_token is not None:
                kwargs["ContinuationToken"] = continuation_token

            response = await self._s3.list_objects_v2(**kwargs)
            for obj in response.get("Contents", []):
                keys.append(obj["Key"])

            if not response.get("IsTruncated"):
                break
            continuation_token = response["NextContinuationToken"]

        return keys

    async def delete_objects(self, bucket: str, keys: Sequence[str]) -> None:
        batch_size = 1000
        for i in range(0, len(keys), batch_size):
            batch = keys[i : i + batch_size]
            await self._s3.delete_objects(
                Bucket=bucket,
                Delete={"Objects": [{"Key": k} for k in batch]},
            )

    async def head_object(self, bucket: str, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            await self._s3.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "404":
                return False
            raise
