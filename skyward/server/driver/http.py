"""HTTP client that mirrors the PoolHost surface over a Unix socket."""
from __future__ import annotations

import json
import struct
from collections.abc import AsyncIterator
from types import TracebackType
from typing import Any

import httpx

from skyward.server.wire import from_dict, to_dict

_HEADERS = {"X-Skyward-Api": "1"}


class ServerClient:
    """Typed wrapper around an ``httpx.AsyncClient`` bound to a UDS.

    Parameters
    ----------
    socket_path
        Absolute path of the Unix domain socket the host is listening on.
        The client issues requests to ``http://skyward`` — the host header
        is cosmetic because the transport is the UDS, not TCP.
    timeout
        Default per-request timeout (seconds). Individual methods can
        override via a ``timeout=`` parameter.
    """

    def __init__(self, socket_path: str, *, timeout: float = 30.0) -> None:
        self._socket_path: str = socket_path
        self._timeout: float = timeout
        transport = httpx.AsyncHTTPTransport(uds=socket_path)
        self._client: httpx.AsyncClient = httpx.AsyncClient(
            transport=transport,
            base_url="http://skyward",
            headers=_HEADERS,
            timeout=timeout,
        )

    async def __aenter__(self) -> ServerClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self._client.aclose()

    @property
    def socket_path(self) -> str:
        """Filesystem path of the bound UDS."""
        return self._socket_path

    async def health(self) -> dict[str, Any]:
        """Return the ``/v1/health`` body."""
        resp = await self._client.get("/v1/health")
        resp.raise_for_status()
        return resp.json()

    async def info(self) -> dict[str, Any]:
        """Return the ``/v1/info`` body (version, pid, python)."""
        resp = await self._client.get("/v1/info")
        resp.raise_for_status()
        return resp.json()

    async def list_compute(self) -> list[Any]:
        """Fetch every compute row the host knows about."""
        resp = await self._client.get("/v1/compute")
        resp.raise_for_status()
        return resp.json()

    async def get_compute(self, name: str) -> Any:
        """Fetch a single compute row; raises on 404."""
        resp = await self._client.get(f"/v1/compute/{name}")
        resp.raise_for_status()
        return resp.json()

    async def ensure_compute(self, name: str, compute_spec: Any) -> Any:
        """Create or resume the named pool via ``POST /v1/compute``.

        Provisioning a real cloud pool can take several minutes, so this
        call disables the read timeout — the server blocks until the
        pool is ``Ready`` (or fails).
        """
        body = {"name": name, "spec": to_dict(compute_spec)}
        resp = await self._client.post("/v1/compute", json=body, timeout=None)
        resp.raise_for_status()
        return resp.json()

    async def shutdown_compute(self, name: str) -> Any:
        """Stop the named pool via ``DELETE /v1/compute/{name}``.

        Teardown can also take a while (instance terminate + volume
        detach), so the read timeout is disabled.
        """
        resp = await self._client.delete(f"/v1/compute/{name}", timeout=None)
        resp.raise_for_status()
        return resp.json()

    async def submit_task(
        self,
        name: str,
        task_key: tuple[str, str],
        payload: bytes,
        timeout: float,
        client_id: str | None = None,
    ) -> tuple[str, bytes]:
        """Submit pickled bytes; return the execution id + pickled reply."""
        headers: dict[str, str] = {
            "X-Task-Module": task_key[0],
            "X-Task-Qualname": task_key[1],
            "X-Timeout": str(timeout),
            "X-Kind": "run",
            "Content-Type": "application/octet-stream",
        }
        if client_id is not None:
            headers["X-Client-Id"] = client_id
        resp = await self._client.post(
            f"/v1/compute/{name}/tasks",
            content=payload,
            headers=headers,
            timeout=timeout + 10,
        )
        resp.raise_for_status()
        execution_id = resp.headers.get("X-Execution-Id", "")
        return execution_id, resp.content

    async def broadcast(
        self,
        name: str,
        task_key: tuple[str, str],
        payload: bytes,
        timeout: float,
        client_id: str | None = None,
    ) -> tuple[str, list[bytes]]:
        """Broadcast and unpack the ``[u32][shard]...`` reply."""
        headers: dict[str, str] = {
            "X-Task-Module": task_key[0],
            "X-Task-Qualname": task_key[1],
            "X-Timeout": str(timeout),
            "X-Kind": "broadcast",
            "Content-Type": "application/octet-stream",
        }
        if client_id is not None:
            headers["X-Client-Id"] = client_id
        resp = await self._client.post(
            f"/v1/compute/{name}/tasks",
            content=payload,
            headers=headers,
            timeout=timeout + 10,
        )
        resp.raise_for_status()
        execution_id = resp.headers.get("X-Execution-Id", "")
        shards = _unpack_broadcast(resp.content)
        return execution_id, shards

    async def subscribe(
        self, name: str, *, since: int = 0,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield SSE events for ``compute:{name}``; closes on EOF."""
        async with self._client.stream(
            "GET", f"/v1/compute/{name}/events",
            params={"since": since},
            timeout=None,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    yield json.loads(line[6:])

    async def subscribe_session_events(
        self, name: str,
    ) -> AsyncIterator[bytes]:
        """Yield pickled :class:`SessionEvent` bytes for ``compute:{name}``.

        Complements :meth:`subscribe` — that one is JSON + persisted
        events; this one streams the live domain events the
        ``SessionProjection`` fires on the host side so the client can
        rebuild UI state identically to in-process mode.
        """
        import base64

        async with self._client.stream(
            "GET", f"/v1/compute/{name}/session-events", timeout=None,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    yield base64.b64decode(line[6:])

    async def node_count(self, name: str) -> int:
        """Return the number of ``ready`` nodes for ``name``."""
        nodes = await self.list_nodes(name)
        return sum(
            1 for n in nodes if n.get("status", {}).get("type") == "NodeReady"
        )

    async def list_nodes(self, name: str) -> list[dict[str, Any]]:
        """Fetch every node row attached to ``compute:{name}``."""
        resp = await self._client.get(f"/v1/compute/{name}/nodes")
        resp.raise_for_status()
        return resp.json()


def _unpack_broadcast(raw: bytes) -> list[bytes]:
    """Inverse of ``routes.executions._pack_broadcast``."""
    out: list[bytes] = []
    pos = 0
    while pos < len(raw):
        (length,) = struct.unpack(">I", raw[pos : pos + 4])
        pos += 4
        out.append(raw[pos : pos + length])
        pos += length
    return out


_ = from_dict  # re-exported for callers composing custom decoders.

__all__ = ["ServerClient"]
