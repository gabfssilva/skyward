"""Persistent cache with decorator for memoization."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any

import cloudpickle

CACHE_DIR = Path.home() / ".skyward" / "cache"
CACHE_VERSION = 1


class DiskCache:
    """Simple disk-based cache with cloudpickle serialization."""

    def __init__(self, namespace: str = "default") -> None:
        self.namespace = namespace
        self.cache_dir = CACHE_DIR / namespace
        self._index: dict[str, datetime] | None = None

    @property
    def index_file(self) -> Path:
        return self.cache_dir / "_index.json"

    @property
    def index(self) -> dict[str, datetime]:
        """Lazy load index (tracks creation times)."""
        if self._index is None:
            self._load_index()
        assert self._index is not None
        return self._index

    def _load_index(self) -> None:
        if self.index_file.exists():
            try:
                raw = json.loads(self.index_file.read_text())
                if raw.get("version") == CACHE_VERSION:
                    self._index = {
                        k: datetime.fromisoformat(v).replace(tzinfo=UTC)
                        for k, v in raw.get("entries", {}).items()
                    }
                else:
                    self._index = {}
            except Exception:
                self._index = {}
        else:
            self._index = {}

    def _save_index(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_file.write_text(
            json.dumps(
                {
                    "version": CACHE_VERSION,
                    "entries": {k: v.isoformat() for k, v in self.index.items()},
                },
                indent=2,
            )
        )

    def _key_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pkl"

    def _make_key(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
        key_data = cloudpickle.dumps((args, kwargs))
        return hashlib.sha256(key_data).hexdigest()[:16]

    def get(self, key: str, ttl: timedelta | None = None) -> tuple[bool, Any]:
        """Get value from cache."""
        if key not in self.index:
            return False, None

        if ttl is not None:
            created = self.index[key]
            if datetime.now(UTC) - created > ttl:
                self._delete(key)
                return False, None

        path = self._key_path(key)
        if not path.exists():
            del self.index[key]
            self._save_index()
            return False, None

        try:
            value = cloudpickle.loads(path.read_bytes())
            return True, value
        except Exception:
            self._delete(key)
            return False, None

    def set(self, key: str, value: Any) -> None:
        """Store value in cache."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._key_path(key).write_bytes(cloudpickle.dumps(value))
        self.index[key] = datetime.now(UTC)
        self._save_index()

    def _delete(self, key: str) -> None:
        path = self._key_path(key)
        if path.exists():
            path.unlink()
        if key in self.index:
            del self.index[key]
            self._save_index()

    def clear(self) -> None:
        """Clear all entries in this namespace."""
        self._index = {}
        if self.cache_dir.exists():
            import shutil

            shutil.rmtree(self.cache_dir)


_caches: dict[str, DiskCache] = {}


def get_cache(namespace: str) -> DiskCache:
    """Get or create cache for namespace."""
    if namespace not in _caches:
        _caches[namespace] = DiskCache(namespace)
    return _caches[namespace]


def cached[**P, R](
    namespace: str = "default",
    ttl: timedelta | None = None,
    key_func: Callable[..., str] | None = None,
    cache_falsy: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator for caching function results to disk."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        cache = get_cache(namespace)

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            cache_args = args[1:] if args and hasattr(args[0], "__dict__") else args

            key = (
                key_func(*args, **kwargs) if key_func else cache._make_key(cache_args, kwargs)
            )

            hit, value = cache.get(key, ttl)
            if hit:
                return value  # type: ignore

            result = func(*args, **kwargs)

            if result is not None and (cache_falsy or result):
                cache.set(key, result)

            return result

        return wrapper

    return decorator
