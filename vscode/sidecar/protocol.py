"""JSON-lines protocol for extension <-> sidecar communication."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Request:
    id: int
    method: str
    params: dict[str, Any]


def parse_request(line: str) -> Request:
    data = json.loads(line)
    if "method" not in data:
        raise ValueError("Missing 'method' field")
    return Request(
        id=data.get("id", 0),
        method=data["method"],
        params=data.get("params", {}),
    )


def format_response(req_id: int, result: Any = None, *, error: str | None = None) -> str:
    if error is not None:
        return json.dumps({"id": req_id, "error": error})
    return json.dumps({"id": req_id, "result": result})


def format_event(event: str, pool: str, data: dict[str, Any]) -> str:
    return json.dumps({"event": event, "pool": pool, "data": data})
