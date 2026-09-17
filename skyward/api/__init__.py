"""The resources the daemon's HTTP API serves, one module per version.

Nothing here imports the rest of skyward: a representation is what a client and the
daemon agree on, and it changes when the API does — not when the model behind it does.
"""

from skyward.api import v1

__all__ = ["v1"]
