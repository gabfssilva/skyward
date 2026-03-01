"""Standalone volume management — upload, download, list, exists, remove."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from concurrent.futures import Future
from contextlib import AbstractAsyncContextManager, suppress
from pathlib import Path
from types import TracebackType
from typing import Any

from skyward.api.provider import ProviderConfig
from skyward.api.spec import Volume
from skyward.providers.provider import Mountable, ObjectStore


class VolumeClient:
    """Sync context manager for standalone CRUD on a Volume's S3 bucket.

    All operations are scoped to the volume's ``prefix``.

    Example
    -------
    >>> vol = Volume(bucket="my-data", mount="/data", prefix="train/")
    >>> with VolumeClient(vol, provider=AWS(region="us-east-1")) as vc:
    ...     vc.upload("./dataset/")
    ...     print(vc.ls())
    ...     vc.download("model.pt", "./local/")
    """

    def __init__(self, volume: Volume, *, provider: ProviderConfig) -> None:
        self._volume = volume
        self._config = provider
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._store: ObjectStore | None = None
        self._store_ctx: AbstractAsyncContextManager[ObjectStore] | None = None
        self._provider: Any = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> VolumeClient:
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="skyward-volume-client",
        )
        self._loop_thread.start()

        try:
            self._run_sync(self._open())
        except Exception:
            self._cleanup()
            raise

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        try:
            self._run_sync(self._close())
        finally:
            self._cleanup()

    async def _open(self) -> None:
        self._provider = await self._config.create_provider()

        if not isinstance(self._provider, Mountable):
            raise TypeError(
                f"Provider {self._config.type!r} does not support volumes. "
                "Supported providers: AWS, GCP, Hyperstack, RunPod.",
            )

        self._store_ctx = self._provider.object_store()
        self._store = await self._store_ctx.__aenter__()

    async def _close(self) -> None:
        if self._store_ctx is not None:
            await self._store_ctx.__aexit__(None, None, None)
            self._store_ctx = None
            self._store = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upload(self, local_path: str | Path, *, key: str | None = None) -> None:
        """Upload a file or directory to the volume.

        Parameters
        ----------
        local_path
            Local file or directory to upload.
        key
            Object key (relative to volume prefix). Defaults to the
            file's basename for files, or the directory's basename for
            directories.
        """
        path = Path(local_path)
        if not path.exists():
            raise FileNotFoundError(f"Local path not found: {path}")

        self._run_sync(self._upload(path, key))

    def download(self, key: str, local_path: str | Path) -> None:
        """Download a file or prefix to a local path.

        Parameters
        ----------
        key
            Object key or prefix (relative to volume prefix).
        local_path
            Local destination path.
        """
        self._run_sync(self._download(key, Path(local_path)))

    def ls(self, prefix: str = "") -> list[str]:
        """List objects under the given prefix.

        Parameters
        ----------
        prefix
            Sub-prefix relative to the volume prefix.

        Returns
        -------
        list[str]
            Object keys relative to the volume prefix.
        """
        return self._run_sync(self._ls(prefix))

    def exists(self, key: str) -> bool:
        """Check if an object exists.

        Parameters
        ----------
        key
            Object key relative to the volume prefix.
        """
        return self._run_sync(self._exists(key))

    def rm(self, key: str) -> None:
        """Remove an object or all objects under a prefix.

        Parameters
        ----------
        key
            Object key or prefix (relative to volume prefix).
        """
        self._run_sync(self._rm(key))

    # ------------------------------------------------------------------
    # Async implementations
    # ------------------------------------------------------------------

    async def _upload(self, path: Path, key: str | None) -> None:
        store = self._get_store()
        bucket = self._volume.bucket

        if path.is_file():
            resolved = self._resolve_key(key or path.name)
            await store.upload_file(bucket, resolved, path)
        else:
            coros = []
            base = path
            for file in path.rglob("*"):
                if not file.is_file():
                    continue
                relative = file.relative_to(base.parent)
                resolved = self._resolve_key(
                    key + "/" + str(file.relative_to(base)) if key else str(relative),
                )
                coros.append(store.upload_file(bucket, resolved, file))
            if coros:
                await asyncio.gather(*coros)

    async def _download(self, key: str, local_path: Path) -> None:
        store = self._get_store()
        bucket = self._volume.bucket
        resolved = self._resolve_key(key)

        if await store.head_object(bucket, resolved):
            local_path.parent.mkdir(parents=True, exist_ok=True)
            await store.download_file(bucket, resolved, local_path)
            return

        keys = await store.list_objects(bucket, resolved)
        if not keys:
            raise FileNotFoundError(f"No objects found for key: {key}")

        coros = []
        for obj_key in keys:
            relative = obj_key.removeprefix(resolved).lstrip("/")
            dest = local_path / relative
            coros.append(store.download_file(bucket, obj_key, dest))
        await asyncio.gather(*coros)

    async def _ls(self, prefix: str) -> list[str]:
        store = self._get_store()
        resolved = self._volume.prefix + prefix
        keys = await store.list_objects(self._volume.bucket, resolved)
        vol_prefix = self._volume.prefix
        return [rel for k in keys if (rel := k.removeprefix(vol_prefix))]

    async def _exists(self, key: str) -> bool:
        store = self._get_store()
        return await store.head_object(self._volume.bucket, self._resolve_key(key))

    async def _rm(self, key: str) -> None:
        store = self._get_store()
        bucket = self._volume.bucket
        resolved = self._resolve_key(key)

        if await store.head_object(bucket, resolved):
            await store.delete_objects(bucket, [resolved])
            return

        keys = await store.list_objects(bucket, resolved)
        if keys:
            await store.delete_objects(bucket, keys)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_key(self, key: str) -> str:
        return self._volume.prefix + key

    def _get_store(self) -> ObjectStore:
        if self._store is None:
            raise RuntimeError("VolumeClient is not open")
        return self._store

    def _run_loop(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_sync[T](self, coro: Coroutine[Any, Any, T], timeout: float = 3600.0) -> T:
        if self._loop is None:
            raise RuntimeError("Event loop not running")
        future: Future[T] = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def _cleanup(self) -> None:
        loop = self._loop
        thread = self._loop_thread
        if loop is None:
            return

        loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=10)

        if not loop.is_running():
            with suppress(Exception):
                loop.close()

        self._loop = None
        self._loop_thread = None
