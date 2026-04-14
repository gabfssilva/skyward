"""Server-Sent Events helper for streaming JSON events to HTTP clients."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from starlette.responses import StreamingResponse

type SseItem = dict[str, Any] | str


async def _stream(gen: AsyncIterator[SseItem]) -> AsyncIterator[bytes]:
    async for item in gen:
        match item:
            case str() as raw:
                yield raw.encode()
            case payload:
                yield f"data: {json.dumps(payload)}\n\n".encode()


def sse_response(gen: AsyncIterator[SseItem]) -> StreamingResponse:
    """Wrap an async generator as a Server-Sent Events streaming response.

    Parameters
    ----------
    gen
        Async iterator yielding either pre-formatted SSE blocks (``str``) or
        JSON-serializable ``dict`` payloads encoded as ``data: <json>\\n\\n``.

    Returns
    -------
    StreamingResponse
        A Starlette response with ``text/event-stream`` media type.
    """
    return StreamingResponse(_stream(gen), media_type="text/event-stream")


__all__ = ["SseItem", "sse_response"]
