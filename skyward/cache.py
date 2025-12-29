"""Persistent cache with decorator for memoization."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

CACHE_DIR = Path.home() / ".skyward" / "cache"
CACHE_VERSION = 1

F = TypeVar("F", bound=Callable[..., Any])


class DiskCache:
    """Simple disk-based cache with JSON serialization."""

    def __init__(self, namespace: str = "default") -> None:
        self.namespace = namespace
        self.cache_file = CACHE_DIR / f"{namespace}.json"
        self._data: dict[str, Any] | None = None

    @property
    def data(self) -> dict[str, Any]:
        """Lazy load cache data."""
        if self._data is None:
            self._load()
        return self._data  # type: ignore

    def _load(self) -> None:
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                raw = json.loads(self.cache_file.read_text())
                if raw.get("version") == CACHE_VERSION:
                    self._data = raw.get("entries", {})
                else:
                    self._data = {}
            except Exception:
                self._data = {}
        else:
            self._data = {}

    def _save(self) -> None:
        """Save cache to disk."""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_text(
            json.dumps({"version": CACHE_VERSION, "entries": self.data}, indent=2)
        )

    def _make_key(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
        """Create cache key from function arguments."""
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    def get(self, key: str, ttl: timedelta | None = None) -> tuple[bool, Any]:
        """Get value from cache.

        Returns:
            Tuple of (hit, value). If hit=False, value is None.
        """
        entry = self.data.get(key)
        if entry is None:
            return False, None

        if ttl is not None:
            created = datetime.fromisoformat(entry["created"])
            if datetime.utcnow() - created > ttl:
                del self.data[key]
                self._save()
                return False, None

        return True, entry["value"]

    def set(self, key: str, value: Any) -> None:
        """Store value in cache."""
        self.data[key] = {
            "value": value,
            "created": datetime.utcnow().isoformat(),
        }
        self._save()

    def clear(self) -> None:
        """Clear all entries in this namespace."""
        self._data = {}
        if self.cache_file.exists():
            self.cache_file.unlink()


_caches: dict[str, DiskCache] = {}


def get_cache(namespace: str) -> DiskCache:
    """Get or create cache for namespace."""
    if namespace not in _caches:
        _caches[namespace] = DiskCache(namespace)
    return _caches[namespace]


def cached(
    namespace: str = "default",
    ttl: timedelta | None = None,
    key_func: Callable[..., str] | None = None,
    cache_falsy: bool = False,
) -> Callable[[F], F]:
    """Decorator for caching function results to disk.

    Args:
        namespace: Cache namespace (separate file per namespace).
        ttl: Optional time-to-live for cache entries.
        key_func: Optional custom function to generate cache key from args.
        cache_falsy: If False (default), don't cache falsy values (False, 0, "", [], etc.).
            This prevents caching "not found" results that may become valid later.

    Example:
        @cached(namespace="aws.ami")
        def find_ami(image_hash: str) -> str:
            return expensive_aws_call(image_hash)
    """

    def decorator(func: F) -> F:
        cache = get_cache(namespace)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Skip 'self' for methods
            cache_args = args[1:] if args and hasattr(args[0], "__dict__") else args

            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = cache._make_key(cache_args, kwargs)

            hit, value = cache.get(key, ttl)
            if hit:
                return value

            result = func(*args, **kwargs)

            # Cache result if it's not None and (cache_falsy or result is truthy)
            if result is not None and (cache_falsy or result):
                cache.set(key, result)

            return result

        return wrapper  # type: ignore

    return decorator
