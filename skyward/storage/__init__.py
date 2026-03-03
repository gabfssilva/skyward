"""Unified S3-compatible storage — dataclass, presets, and context-manager CRUD."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from concurrent.futures import Future
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

from .presets import GCS, R2, S3, Backblaze, Wasabi

_STATE: dict[int, _StoreState] = {}


@dataclass
class _StoreState:
    loop: asyncio.AbstractEventLoop
    loop_thread: threading.Thread
    session: Any
    s3_ctx: Any
    s3: Any


@dataclass(frozen=True, slots=True)
class Storage:
    """S3-compatible object storage endpoint.

    Parameters
    ----------
    endpoint
        Full HTTPS endpoint URL for the S3-compatible service.
    access_key
        Access key ID. ``None`` defers to environment / IAM credentials.
    secret_key
        Secret access key. ``None`` defers to environment / IAM credentials.
    path_style
        Use path-style addressing instead of virtual-hosted-style.
    """

    endpoint: str
    access_key: str | None = None
    secret_key: str | None = None
    path_style: bool = False

    def __enter__(self) -> Storage:
        loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=_run_loop,
            args=(loop,),
            daemon=True,
            name="skyward-storage",
        )
        thread.start()

        try:
            _run(loop, _open_store(self, loop, thread))
        except Exception:
            _cleanup(self)
            raise

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        try:
            state = _STATE.get(id(self))
            if state is not None and state.s3_ctx is not None:
                _run(state.loop, state.s3_ctx.__aexit__(None, None, None))
        finally:
            _cleanup(self)

    def upload(self, bucket: str, local_path: str | Path, *, key: str | None = None) -> None:
        """Upload a file or directory to a bucket.

        Parameters
        ----------
        bucket
            Target S3 bucket name.
        local_path
            Local file or directory to upload.
        key
            Object key. Defaults to the file's basename.
        """
        path = Path(local_path)
        if not path.exists():
            raise FileNotFoundError(f"Local path not found: {path}")
        state = _get_state(self)
        _run(state.loop, _upload(state.s3, bucket, path, key))

    def download(self, bucket: str, key: str, local_path: str | Path) -> None:
        """Download an object to a local path.

        Parameters
        ----------
        bucket
            Source S3 bucket name.
        key
            Object key to download.
        local_path
            Local destination path.
        """
        state = _get_state(self)
        _run(state.loop, _download(state.s3, bucket, key, Path(local_path)))

    def ls(self, bucket: str, prefix: str = "") -> list[str]:
        """List objects in a bucket.

        Parameters
        ----------
        bucket
            S3 bucket name.
        prefix
            Key prefix filter.

        Returns
        -------
        list[str]
            Matching object keys.
        """
        state = _get_state(self)
        return _run(state.loop, _ls(state.s3, bucket, prefix))

    def exists(self, bucket: str, key: str) -> bool:
        """Check if an object exists.

        Parameters
        ----------
        bucket
            S3 bucket name.
        key
            Object key.
        """
        state = _get_state(self)
        return _run(state.loop, _exists(state.s3, bucket, key))

    def rm(self, bucket: str, key: str) -> None:
        """Remove an object or all objects under a prefix.

        Parameters
        ----------
        bucket
            S3 bucket name.
        key
            Object key or prefix to remove.
        """
        state = _get_state(self)
        _run(state.loop, _rm(state.s3, bucket, key))


def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


def _run[T](loop: asyncio.AbstractEventLoop, coro: Coroutine[Any, Any, T], timeout: float = 3600.0) -> T:
    future: Future[T] = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=timeout)


def _get_state(storage: Storage) -> _StoreState:
    state = _STATE.get(id(storage))
    if state is None:
        raise RuntimeError("Storage is not open — use as a context manager")
    return state


async def _open_store(storage: Storage, loop: asyncio.AbstractEventLoop, thread: threading.Thread) -> None:
    import aioboto3

    kwargs: dict[str, Any] = {"endpoint_url": storage.endpoint}
    if storage.access_key is not None:
        kwargs["aws_access_key_id"] = storage.access_key
    if storage.secret_key is not None:
        kwargs["aws_secret_access_key"] = storage.secret_key

    session = aioboto3.Session()
    s3_ctx = session.client("s3", **kwargs)
    s3 = await s3_ctx.__aenter__()  # pyright: ignore[reportAttributeAccessIssue]

    _STATE[id(storage)] = _StoreState(
        loop=loop,
        loop_thread=thread,
        session=session,
        s3_ctx=s3_ctx,
        s3=s3,
    )


def _cleanup(storage: Storage) -> None:
    state = _STATE.pop(id(storage), None)
    if state is None:
        return

    state.loop.call_soon_threadsafe(state.loop.stop)
    state.loop_thread.join(timeout=10)

    if not state.loop.is_running():
        with suppress(Exception):
            state.loop.close()


async def _upload(s3: Any, bucket: str, path: Path, key: str | None) -> None:
    if path.is_file():
        resolved = key or path.name
        content = path.read_bytes()
        await s3.put_object(Bucket=bucket, Key=resolved, Body=content, ContentLength=len(content))
    else:
        coros = []
        for file in path.rglob("*"):
            if not file.is_file():
                continue
            relative = file.relative_to(path.parent)
            resolved = key + "/" + str(file.relative_to(path)) if key else str(relative)
            content = file.read_bytes()
            coros.append(s3.put_object(Bucket=bucket, Key=resolved, Body=content, ContentLength=len(content)))
        if coros:
            await asyncio.gather(*coros)


async def _download(s3: Any, bucket: str, key: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    await s3.download_file(bucket, key, str(local_path))


async def _ls(s3: Any, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    continuation_token: str | None = None

    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if continuation_token is not None:
            kwargs["ContinuationToken"] = continuation_token

        response = await s3.list_objects_v2(**kwargs)
        for obj in response.get("Contents", []):
            keys.append(obj["Key"])

        if not response.get("IsTruncated"):
            break
        continuation_token = response["NextContinuationToken"]

    return keys


async def _exists(s3: Any, bucket: str, key: str) -> bool:
    from botocore.exceptions import ClientError

    try:
        await s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "404":
            return False
        raise


async def _rm(s3: Any, bucket: str, key: str) -> None:
    from botocore.exceptions import ClientError

    try:
        await s3.head_object(Bucket=bucket, Key=key)
        await s3.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": key}]})
        return
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") != "404":
            raise

    keys = await _ls(s3, bucket, key)
    if keys:
        batch_size = 1000
        for i in range(0, len(keys), batch_size):
            batch = keys[i : i + batch_size]
            await s3.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": k} for k in batch]})


__all__ = [
    "Storage",
    "R2",
    "S3",
    "GCS",
    "Wasabi",
    "Backblaze",
]
