"""UUIDv7 generator.

Python 3.14 ships :func:`uuid.uuid7`; on 3.12/3.13 we implement the
format per RFC 9562 §5.7 — 48-bit ``unix_ts_ms`` in the top, 4 version
bits, 12 random ``rand_a`` bits, 2 variant bits, and 62 random
``rand_b`` bits.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from uuid import UUID

if sys.version_info >= (3, 14):
    from uuid import uuid7 as _stdlib_uuid7

    def uuid7() -> UUID:
        """Return a fresh UUIDv7 using the stdlib generator."""
        return _stdlib_uuid7()
else:
    _monotonic_lock = threading.Lock()
    _last_ms = 0
    _last_seq = 0

    def uuid7() -> UUID:
        """Return a fresh UUIDv7 with intra-millisecond monotonicity.

        Within the same millisecond the low 12 ``rand_a`` bits carry a
        counter so consecutive ids sort strictly ascending, matching the
        RFC 9562 §6.2 "Method 1: Fixed-Length Dedicated Counter Bits".
        """
        global _last_ms, _last_seq

        with _monotonic_lock:
            ts_ms = int(time.time() * 1000) & ((1 << 48) - 1)
            if ts_ms <= _last_ms:
                ts_ms = _last_ms
                _last_seq = (_last_seq + 1) & 0x0FFF
                if _last_seq == 0:
                    ts_ms = _last_ms + 1
            else:
                _last_seq = int.from_bytes(os.urandom(2), "big") & 0x0FFF
            _last_ms = ts_ms
            rand_a = _last_seq

        rand_b = int.from_bytes(os.urandom(8), "big") & ((1 << 62) - 1)
        value = ts_ms << 80
        value |= 0x7 << 76
        value |= rand_a << 64
        value |= 0b10 << 62
        value |= rand_b
        return UUID(int=value)


__all__ = ["uuid7"]
