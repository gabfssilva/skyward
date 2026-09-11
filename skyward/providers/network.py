"""The TLS context every httpx-based adapter shares.

Each ``httpx.AsyncClient`` builds its own SSL context by default, which reads
and parses the CA bundle on the event loop — milliseconds per client, and
adapters open a client per call. One context, built once with the same trust
httpx would use (certifi and the ``SSL_CERT_*`` environment), is handed to
every client instead.
"""

import ssl
from functools import cache

import httpx


@cache
def tls() -> ssl.SSLContext:
    return httpx.create_ssl_context()


__all__ = ["tls"]
