"""Entry point: uv run python -m vscode.sidecar"""

from __future__ import annotations

import asyncio
import json
import sys

from .bridge import Bridge
from .protocol import format_response, parse_request


async def main() -> None:
    bridge = Bridge()

    def emit(line: str) -> None:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    bridge.set_event_callback(emit)

    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)

    while True:
        raw = await reader.readline()
        if not raw:
            break
        line = raw.decode().strip()
        if not line:
            continue

        try:
            req = parse_request(line)
            result = await bridge.handle(req.method, req.params)
            emit(format_response(req.id, result))
        except Exception as exc:
            try:
                req_id = json.loads(line).get("id", 0)
            except Exception:
                req_id = 0
            emit(format_response(req_id, error=str(exc)))

    await bridge.close()


if __name__ == "__main__":
    asyncio.run(main())
