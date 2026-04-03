"""Bridge between JSON-lines (extension) and daemon (cloudpickle/Unix socket)."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from skyward.api.events import Log
from skyward.api.views import SessionView
from skyward.daemon.client import DaemonClient
from skyward.daemon.protocol import (
    GetPools,
    GetPoolView,
    PoolList,
    PoolViewResponse,
)

from .protocol import format_event
from .serialize import event_to_dict, pool_view_to_dict


class Bridge:
    """Stateful bridge managing daemon connections and event subscriptions."""

    def __init__(self) -> None:
        self._client: DaemonClient | None = None
        self._subscriptions: dict[str, asyncio.Task[None]] = {}
        self._log_buffers: dict[str, deque[dict[str, Any]]] = {}
        self._event_callback: Callable[[str], None] | None = None

    def set_event_callback(self, cb: Callable[[str], None]) -> None:
        self._event_callback = cb

    async def _connect(self) -> DaemonClient:
        if self._client is not None:
            return self._client
        client = DaemonClient()
        await client.connect()
        self._client = client
        return client

    async def handle(self, method: str, params: dict[str, Any]) -> Any:
        match method:
            case "daemon/ping":
                return await self._ping()
            case "daemon/start":
                return await self._start_daemon()
            case "pools/list":
                return await self._list_pools()
            case "pools/view":
                return await self._get_pool_view(params["pool"])
            case "pools/ensure":
                return await self._ensure_pool(params["name"], params.get("project_dir"))
            case "pools/shutdown":
                return await self._shutdown_pool(params["pool"])
            case "pools/subscribe":
                self._subscribe(params["pool"])
                return {"ok": True}
            case "pools/unsubscribe":
                self._unsubscribe(params["pool"])
                return {"ok": True}
            case "config/pools":
                return self._config_pools()
            case "config/providers":
                return self._config_providers()
            case "discover/functions":
                return self._discover_functions(params.get("files"))
            case "run/main":
                return await self._run_main(params["file"], params["fn"], params["args"], params["pool"])
            case _:
                raise ValueError(f"Unknown method: {method}")

    async def _ping(self) -> dict[str, Any]:
        try:
            client = await self._connect()
            await client.ping()
            return {"ok": True}
        except Exception:
            return {"ok": False}

    async def _start_daemon(self) -> dict[str, Any]:
        from skyward.daemon.spawn import ensure_daemon, is_daemon_running

        if is_daemon_running():
            return {"ok": True, "already_running": True}
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, ensure_daemon)
        return {"ok": True, "already_running": False}

    async def _list_pools(self) -> dict[str, Any]:
        try:
            client = await self._connect()
            resp = await client.request(GetPools())
            match resp:
                case PoolList(pools=pools):
                    return {"pools": [
                        {
                            "name": p.name,
                            "phase": p.phase,
                            "ready_nodes": p.nodes_ready,
                            "total_nodes": p.nodes_total,
                        }
                        for p in pools
                    ]}
            return {"pools": []}
        except Exception:
            return {"pools": []}

    async def _get_pool_view(self, pool: str) -> dict[str, Any]:
        client = await self._connect()
        resp = await client.request(GetPoolView(pool_name=pool))
        match resp:
            case PoolViewResponse(view=view):
                logs = list(self._log_buffers.get(pool, []))
                return pool_view_to_dict(view, logs=logs)
        raise RuntimeError(f"Pool '{pool}' not found")

    async def _ensure_pool(self, name: str, project_dir: str | None) -> dict[str, Any]:
        client = await self._connect()
        await client.ensure_pool(name)
        return {"ok": True}

    async def _shutdown_pool(self, pool: str) -> dict[str, Any]:
        client = await self._connect()
        await client.shutdown_pool(pool)
        self._unsubscribe(pool)
        return {"ok": True}

    def _subscribe(self, pool: str) -> None:
        if pool in self._subscriptions:
            return
        self._log_buffers.setdefault(pool, deque(maxlen=200))
        task = asyncio.create_task(self._event_loop(pool))
        self._subscriptions[pool] = task

    def _unsubscribe(self, pool: str) -> None:
        task = self._subscriptions.pop(pool, None)
        if task:
            task.cancel()

    async def _event_loop(self, pool: str) -> None:
        try:
            client = DaemonClient()
            await client.connect()
            async for msg in client.subscribe(pool):
                match msg:
                    case SessionView():
                        pass
                    case Log.Emitted():
                        entry = {
                            "ts": time.time(),
                            "level": msg.level,
                            "node_id": msg.node_id,
                            "message": msg.message,
                        }
                        self._log_buffers.setdefault(pool, deque(maxlen=200)).append(entry)
                        if self._event_callback:
                            self._event_callback(format_event("log.emitted", pool, entry))
                    case _:
                        d = event_to_dict(msg)
                        if d and self._event_callback:
                            self._event_callback(format_event(d["event"], d["pool"], d["data"]))
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        finally:
            self._subscriptions.pop(pool, None)

    def _config_pools(self) -> dict[str, Any]:
        from pathlib import Path

        from skyward.config import load_config

        config = load_config(project_dir=Path.cwd())
        pools = config.get("pools", {}) if isinstance(config, dict) else {}
        return {"pools": list(pools.keys())}

    def _config_providers(self) -> dict[str, Any]:
        from pathlib import Path

        from skyward.config import load_config

        config = load_config(project_dir=Path.cwd())
        providers = config.get("providers", {}) if isinstance(config, dict) else {}
        return {"providers": list(providers.keys())}

    def _discover_functions(self, files: list[str] | None) -> dict[str, Any]:
        from .discovery import discover_main_functions

        return {"functions": discover_main_functions(files)}

    async def _run_main(self, file: str, fn: str, args: dict[str, Any], pool: str) -> dict[str, Any]:
        from .executor import run_main

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, run_main, file, fn, args, pool)
        return {"ok": True}

    async def close(self) -> None:
        for task in self._subscriptions.values():
            task.cancel()
        self._subscriptions.clear()
        if self._client:
            await self._client.close()
            self._client = None
