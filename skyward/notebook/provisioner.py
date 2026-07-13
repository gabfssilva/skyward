"""Skyward remote-kernel provisioner for ``jupyter_client``.

Runs inside the local Jupyter process. Opens its own asyncssh connection to
the session's head node, launches ``ipykernel`` inside ``/opt/skyward/.venv``,
reads back the kernel's freshly chosen ports + HMAC key, and forwards the 5
ZMQ channels through local ephemeral ports. Nothing kernel-related transits
the Skyward server or Casty.

Co-located, key-auth only: the laptop must reach the node directly and read
the SSH private key locally (mirrors ``sky console``/``sky repl``).
"""

from __future__ import annotations

import asyncio
import json
import shlex
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
from jupyter_client.connect import KernelConnectionInfo
from jupyter_client.provisioning.provisioner_base import KernelProvisionerBase
from traitlets import Float, Unicode

from . import _client

__all__ = ["SkywardKernelProvisioner"]

_CHANNELS = ("shell_port", "iopub_port", "stdin_port", "control_port", "hb_port")


class SkywardKernelProvisioner(KernelProvisionerBase):
    """Launch and manage an ``ipykernel`` on a Skyward head node over SSH."""

    session = Unicode(config=True)
    url = Unicode("http://localhost:7590", config=True)
    poll_interval = Float(2.0, config=True)
    poll_timeout = Float(600.0, config=True)

    def __init__(self, connect_fn: Callable[[], Awaitable[Any]] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._connect_fn = connect_fn
        self._conn: Any = None
        self._proc: Any = None
        self._listeners: list[Any] = []

    @property
    def has_process(self) -> bool:
        """Whether a remote kernel process handle is currently held."""
        return self._proc is not None

    async def pre_launch(self, **kwargs: Any) -> dict[str, Any]:
        """Provision the SSH tunnel and remote kernel, returning launch kwargs.

        Idempotent across restarts: an open connection is reused and stale
        forward listeners are closed before re-forwarding.
        """
        kwargs = await super().pre_launch(**kwargs)

        await self._ensure_pool()

        node = await self._fetch_head_node()
        key_path = node.get("ssh_key_path")
        if not key_path or not Path(key_path).exists():
            if node.get("has_password"):
                raise RuntimeError(
                    "password-only nodes are not supported for the remote kernel (Phase 1)",
                )
            raise RuntimeError(
                "the remote kernel requires the client to run on the same host "
                "as the server (Phase 1 limitation)",
            )

        ssh_user = node["ssh_user"]
        connect = self._connect_fn or _client.default_connect(
            node["ip"], node["ssh_port"], ssh_user, key_path,
        )
        if self._conn is None:
            self._conn = await connect()

        self._close_listeners()
        await self._ensure_ipykernel(ssh_user)

        remote_file = f"/opt/skyward/kernel-{self.kernel_id}.json"
        launch = self._launch_command(ssh_user, remote_file)
        self._proc = await self._conn.create_process(launch, encoding=None)

        raw = await _client.read_remote_file(self._conn, remote_file)
        self._remote = json.loads(raw)

        self._local_ports = {}
        for channel in _CHANNELS:
            local, listener = await _client.forward_local_port(self._conn, self._remote[channel])
            self._local_ports[channel] = local
            self._listeners.append(listener)

        kwargs["cmd"] = shlex.split(launch)
        return kwargs

    async def launch_kernel(self, cmd: list[str], **kwargs: Any) -> KernelConnectionInfo:
        """Return the assembled local connection info for the running kernel.

        The remote process is already started in :meth:`pre_launch`; ``cmd`` is
        informational.
        """
        key = self._remote["key"]
        info: KernelConnectionInfo = {
            "ip": "127.0.0.1",
            "transport": "tcp",
            "signature_scheme": "hmac-sha256",
            "shell_port": self._local_ports["shell_port"],
            "iopub_port": self._local_ports["iopub_port"],
            "stdin_port": self._local_ports["stdin_port"],
            "control_port": self._local_ports["control_port"],
            "hb_port": self._local_ports["hb_port"],
        }
        info["key"] = key.encode() if isinstance(key, str) else key  # type: ignore[typeddict-item]  # upstream types key as str; ipykernel stores bytes
        self.connection_info = info
        return info

    async def poll(self) -> int | None:
        """Return the remote process exit code, or ``None`` while alive."""
        return None if self._proc is None else self._proc.returncode

    async def wait(self) -> int | None:
        """Await remote process termination and return its exit code."""
        if self._proc is None:
            return None
        result = await self._proc.wait()
        return getattr(result, "exit_status", 0) or 0

    async def send_signal(self, signum: int) -> None:
        """Forward a signal to the remote process."""
        if self._proc is not None:
            self._proc.send_signal(signum)

    async def terminate(self, restart: bool = False) -> None:
        """Terminate the remote process (SIGTERM)."""
        if self._proc is not None:
            self._proc.terminate()

    async def kill(self, restart: bool = False) -> None:
        """Kill the remote process (SIGKILL)."""
        if self._proc is not None:
            self._proc.kill()

    async def cleanup(self, restart: bool = False) -> None:
        """Close forward listeners; close the SSH connection unless restarting."""
        self._close_listeners()
        if not restart and self._conn is not None:
            self._conn.close()
            self._conn = None

    async def _ensure_pool(self) -> None:
        """Ensure the session's pool exists and is ready before tunnelling.

        Reuses a pool that is already ``ready`` with nodes. On 404 it resolves
        the session's named pool from ``skyward.toml`` and creates it via
        ``POST /compute?name=`` (treating 409 as a lost creation race), then
        polls readiness. Raises if no matching named pool exists, if the pool
        is scaled to zero, or if provisioning exceeds :attr:`poll_timeout`.
        """
        async with httpx.AsyncClient(base_url=self.url) as client:
            resp = await client.get(f"/compute/{self.session}")
            if resp.status_code == 200:
                info = resp.json()
                self._check_ready(info)
                if info.get("status") == "ready":
                    return
            elif resp.status_code == 404:
                await self._create_pool(client)

            await self._poll_ready(client)

    def _check_ready(self, info: dict[str, Any]) -> None:
        if info.get("status") == "ready" and info.get("current_nodes", 0) == 0:
            raise RuntimeError(
                f"pool {self.session!r} is scaled to zero; waking a slept pool "
                "is not supported yet. Recreate or resize it: "
                f"sky compute resize {self.session} --nodes 1",
            )

    async def _create_pool(self, client: httpx.AsyncClient) -> None:
        from skyward.config import resolve_pool_specs
        from skyward.server.wire import encode

        try:
            specs, options = resolve_pool_specs(self.session)
        except (KeyError, ValueError) as exc:
            raise RuntimeError(
                f"session {self.session!r} does not exist on the server and no "
                f"matching named pool was found in skyward.toml ({exc}). "
                f"Create it first: sky new {self.session}  — or add "
                f"[pool.{self.session}] to skyward.toml",
            ) from None

        body = encode((specs, options))
        resp = await client.post(
            "/compute",
            params={"name": self.session},
            content=body,
            headers={"Content-Type": "application/octet-stream"},
        )
        if resp.status_code == 409:
            return
        resp.raise_for_status()

    async def _poll_ready(self, client: httpx.AsyncClient) -> None:
        try:
            async with asyncio.timeout(self.poll_timeout):
                while True:
                    resp = await client.get(f"/compute/{self.session}")
                    if resp.status_code == 200:
                        info = resp.json()
                        if info.get("status") == "ready" and info.get("current_nodes", 0) > 0:
                            return
                        if info.get("status") == "failed":
                            raise RuntimeError(
                                f"pool {self.session!r} failed to provision: "
                                f"{info.get('error')}",
                            )
                    await asyncio.sleep(self.poll_interval)
        except TimeoutError:
            raise RuntimeError(
                f"pool {self.session!r} did not become ready within "
                f"{self.poll_timeout:.0f}s",
            ) from None

    async def _fetch_head_node(self) -> dict[str, Any]:
        async with httpx.AsyncClient(base_url=self.url) as client:
            resp = await client.get(f"/compute/{self.session}/nodes")
        resp.raise_for_status()
        nodes = resp.json()["nodes"]
        node = next((n for n in nodes if n["rank"] == 0), None)
        if node is None:
            raise RuntimeError(f"session {self.session!r} has no head node")
        return node

    async def _ensure_ipykernel(self, ssh_user: str) -> None:
        uv = "$(command -v uv || echo /root/.local/bin/uv)"
        inner = f"cd /opt/skyward && {uv} add ipykernel"
        cmd = inner if ssh_user == "root" else f"sudo bash -c {shlex.quote(inner)}"
        code, _, stderr = await _client.run(self._conn, cmd)
        if code != 0:
            raise RuntimeError(f"failed to install ipykernel (exit {code}): {stderr}")

    def _launch_command(self, ssh_user: str, remote_file: str) -> str:
        inner = (
            f"cd /opt/skyward && exec /opt/skyward/.venv/bin/python "
            f"-m ipykernel_launcher -f {remote_file}"
        )
        if ssh_user == "root":
            return inner
        return f"sudo bash -c {shlex.quote(inner)}"

    def _close_listeners(self) -> None:
        for listener in self._listeners:
            listener.close()
        self._listeners = []
