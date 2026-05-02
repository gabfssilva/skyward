"""Shared HTTP-server helpers for the Skyward CLI.

URL resolution order: ``--url`` flag > ``SKYWARD_SERVER_URL`` env > the
default ``http://localhost:7590`` exported by :mod:`skyward.server.client`.
"""

from __future__ import annotations

import os

import httpx

from skyward.server.client import DEFAULT_URL

DEFAULT_TIMEOUT = httpx.Timeout(30.0, read=None)


def resolve_server_url(url: str | None) -> str:
    if url:
        return url.rstrip("/")
    if env := os.environ.get("SKYWARD_SERVER_URL"):
        return env.rstrip("/")
    return DEFAULT_URL


def make_client(url: str | None) -> httpx.Client:
    return httpx.Client(base_url=resolve_server_url(url), timeout=DEFAULT_TIMEOUT)


def format_http_error(r: httpx.Response) -> str:
    try:
        body = r.json()
    except ValueError:
        body = r.text or r.reason_phrase
    if isinstance(body, dict) and "error" in body:
        return f"HTTP {r.status_code}: {body['error']}"
    return f"HTTP {r.status_code}: {body}"
