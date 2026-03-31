"""VS Code sidecar bridge — JSON-lines protocol over stdio."""
from __future__ import annotations

from .bridge import Bridge
from .protocol import Request, format_event, format_response, parse_request
from .serialize import event_to_dict, pool_view_to_dict

__all__ = [
    "Bridge",
    "Request",
    "event_to_dict",
    "format_event",
    "format_response",
    "parse_request",
    "pool_view_to_dict",
]
