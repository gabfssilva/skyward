"""HTTP client that attaches to a remote pool exposed by ``skyward.server``.

``sky.Client(name="my-pool")`` returns an object that satisfies the public
``Pool`` operator surface (``>>``, ``@``, ``>``) by proxying every dispatch
to the configured server.  The pool must already exist remotely — the
client never creates or destroys pools.
"""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from types import TracebackType
from typing import TYPE_CHECKING, Any

import httpx

from skyward.server.wire import decode, encode

if TYPE_CHECKING:
    from skyward.api.function import PendingFunction


DEFAULT_URL = "http://localhost:7590"
_POLL_INITIAL = 0.1
_POLL_MAX = 2.0


class Client:
    """HTTP client attached to a named pool on a Skyward server.

    Parameters
    ----------
    name
        Name of the remote pool.  Must already exist on the server.
    url
        Base URL of the Skyward server.  Defaults to ``http://localhost:8000``.

    Examples
    --------
    >>> with sky.Client(name="train") as compute:
    ...     result = train(data) >> compute
    """

    def __init__(self, name: str, *, url: str = DEFAULT_URL) -> None:
        self.name = name
        self.url = url.rstrip("/")
        self._http = httpx.Client(base_url=self.url, timeout=httpx.Timeout(30.0, read=None))
        self._poll_executor = ThreadPoolExecutor(
            max_workers=8, thread_name_prefix=f"sky-client-{name}",
        )

    def __enter__(self) -> Client:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        del exc_type, exc_val, exc_tb
        self.close()

    def close(self) -> None:
        self._poll_executor.shutdown(wait=False, cancel_futures=True)
        self._http.close()

    @property
    def is_active(self) -> bool:
        try:
            return bool(self._info()["is_active"])
        except Exception:
            return False

    @property
    def concurrency(self) -> int:
        return int(self._info()["concurrency"])

    def current_nodes(self) -> int:
        return int(self._info()["current_nodes"])

    def run[T](self, pending: PendingFunction[T]) -> T:
        eid = self._submit(pending, mode="run")
        return self._wait(eid)

    def run_async[T](self, pending: PendingFunction[T]) -> Future[T]:
        eid = self._submit(pending, mode="run")
        return self._poll_executor.submit(self._wait, eid)

    def broadcast[T](self, pending: PendingFunction[T]) -> list[T]:
        eid = self._submit(pending, mode="broadcast")
        return self._wait(eid)

    def _info(self) -> dict:
        r = self._http.get(f"/compute/{self.name}")
        if r.status_code == 404:
            raise RuntimeError(f"pool {self.name!r} does not exist on {self.url}")
        r.raise_for_status()
        return r.json()

    def _submit(self, pending: PendingFunction, mode: str) -> str:
        body = encode(pending)
        r = self._http.post(
            f"/compute/{self.name}/executions",
            params={"mode": mode},
            content=body,
            headers={"Content-Type": "application/octet-stream"},
        )
        if r.status_code == 404:
            raise RuntimeError(f"pool {self.name!r} not found on {self.url}")
        r.raise_for_status()
        return r.json()["id"]

    def _wait(self, eid: str) -> Any:
        backoff = _POLL_INITIAL
        url = f"/compute/{self.name}/executions/{eid}"
        while True:
            r = self._http.get(url)
            match r.status_code:
                case 202:
                    time.sleep(backoff)
                    backoff = min(backoff * 1.5, _POLL_MAX)
                case 200:
                    return decode(r.content)
                case 500 if r.headers.get("X-Skyward-Error") == "1":
                    raise decode(r.content)
                case 404:
                    raise RuntimeError(f"execution {eid} disappeared")
                case _:
                    r.raise_for_status()
                    raise RuntimeError(f"unexpected status {r.status_code}")
