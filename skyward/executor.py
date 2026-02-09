"""HTTP-based executor for remote function execution via Casty.

Submits jobs to the Casty worker's HTTP API on the head node.
Uses an SSH tunnel to forward requests to the remote HTTP server.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

import asyncssh
import httpx
from loguru import logger

from .utils.serialization import deserialize, serialize

CASTY_PORT = 25520
HTTP_PORT = 8265


@dataclass
class Executor:
    head_ip: str
    user: str
    key_path: str
    num_nodes: int = 1
    ssh_port: int = 22
    http_port: int = HTTP_PORT
    connect_timeout: float = 120.0
    job_timeout: float = 600.0
    env_vars: dict[str, str] = field(default_factory=dict)
    pool_infos: list[str] = field(default_factory=list)

    _ssh_conn: asyncssh.SSHClientConnection | None = field(default=None, repr=False)
    _tunnel_listener: Any = field(default=None, repr=False)
    _http_client: httpx.AsyncClient | None = field(default=None, repr=False)
    _connected: bool = field(default=False, repr=False)

    async def connect(self) -> None:
        if self._connected:
            return

        logger.debug(f"Connecting SSH to {self.head_ip}:{self.ssh_port}")
        self._ssh_conn = await asyncssh.connect(
            self.head_ip,
            port=self.ssh_port,
            username=self.user,
            client_keys=[self.key_path],
            known_hosts=None,
            connect_timeout=self.connect_timeout,
            keepalive_interval=15,
            keepalive_count_max=4,
        )

        logger.debug(f"Opening SSH tunnel to {self.head_ip}:{self.http_port}")
        self._tunnel_listener = await self._ssh_conn.forward_local_port(
            "", 0, "127.0.0.1", self.http_port,
        )
        local_port = self._tunnel_listener.get_port()

        self._http_client = httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{local_port}",
            timeout=httpx.Timeout(self.job_timeout, connect=30.0),
        )

        await self._wait_for_ready()

        self._connected = True
        logger.info(f"Connected to Casty HTTP API at {self.head_ip}:{self.http_port}")

    async def _wait_for_ready(self) -> None:
        assert self._http_client is not None

        for attempt in range(30):
            try:
                resp = await self._http_client.get("/health")
                if resp.status_code == 200:
                    logger.debug("Casty HTTP API is ready")
                    return
            except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError):
                pass

            if attempt < 29:
                logger.debug(f"Casty HTTP not ready (attempt {attempt + 1}/30)")
                await asyncio.sleep(2.0)

        raise RuntimeError(f"Casty HTTP API not ready after 60s at {self.head_ip}:{self.http_port}")

    async def disconnect(self) -> None:
        logger.debug(f"Disconnecting from {self.head_ip}")
        self._connected = False

        if self._http_client is not None:
            with suppress(Exception):
                await self._http_client.aclose()
            self._http_client = None

        if self._tunnel_listener is not None:
            with suppress(Exception):
                self._tunnel_listener.close()
            self._tunnel_listener = None

        if self._ssh_conn is not None:
            with suppress(Exception):
                self._ssh_conn.close()
            with suppress(Exception):
                await asyncio.wait_for(self._ssh_conn.wait_closed(), timeout=5.0)
            self._ssh_conn = None

        logger.debug(f"Disconnected from {self.head_ip}")

    async def __aenter__(self) -> Executor:
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.disconnect()

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def execute[T](
        self,
        fn: Callable[..., T],
        *args: Any,
        node_id: int | None = None,
        **kwargs: Any,
    ) -> T:
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")

        assert self._http_client is not None
        fn_name = getattr(fn, "__name__", str(fn))
        logger.debug(f"execute({fn_name}) node_id={node_id}")

        payload = serialize({
            "fn": fn,
            "args": args,
            "kwargs": kwargs,
            "node_id": node_id,
        })

        resp = await self._http_client.post(
            "/jobs",
            content=payload,
            headers={"Content-Type": "application/octet-stream"},
        )

        result = deserialize(resp.content)

        if resp.status_code != 200:
            error = result.get("error", "Unknown error") if isinstance(result, dict) else str(result)
            tb = result.get("traceback", "") if isinstance(result, dict) else ""
            raise RuntimeError(f"Execution error: {error}\n{tb}")

        logger.debug(f"execute({fn_name}) completed")
        return result

    async def broadcast[T](
        self,
        fn: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> list[T]:
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")

        assert self._http_client is not None
        fn_name = getattr(fn, "__name__", str(fn))
        logger.debug(f"broadcast({fn_name})")

        payload = serialize({
            "fn": fn,
            "args": args,
            "kwargs": kwargs,
            "num_nodes": self.num_nodes,
        })

        resp = await self._http_client.post(
            "/jobs/broadcast",
            content=payload,
            headers={"Content-Type": "application/octet-stream"},
        )

        result = deserialize(resp.content)

        if resp.status_code != 200:
            error = result.get("error", "Unknown error") if isinstance(result, dict) else str(result)
            tb = result.get("traceback", "") if isinstance(result, dict) else ""
            raise RuntimeError(f"Broadcast error: {error}\n{tb}")

        logger.debug(f"broadcast({fn_name}) completed")
        return result

    async def setup_cluster(
        self,
        env_vars: dict[str, str],
    ) -> None:
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")

        def setup_env(all_pool_infos: list[str], extra_vars: dict[str, str]) -> str:
            import os
            nid = int(os.environ.get("SKYWARD_NODE_ID", "0"))
            if all_pool_infos and nid < len(all_pool_infos):
                os.environ["COMPUTE_POOL"] = all_pool_infos[nid]
            for key, value in extra_vars.items():
                os.environ[key] = value
            return "ok"

        await self.broadcast(setup_env, self.pool_infos, env_vars)


__all__ = [
    "Executor",
    "CASTY_PORT",
    "HTTP_PORT",
]
