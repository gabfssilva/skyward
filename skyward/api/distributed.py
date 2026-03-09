"""Distributed collection types shared across the public API.

Defines consistency-level options for distributed data structures
(dict, set, counter) that are synchronized across pool nodes.
"""

from __future__ import annotations

from typing import Literal

type Consistency = Literal["strong", "eventual"]
"""Consistency guarantee for distributed collections.

- ``"strong"`` — linearizable reads and writes.  Every operation is
  forwarded to the head node and applied in total order.  Highest
  correctness, but every operation incurs a network round-trip.
- ``"eventual"`` — reads may return stale values.  Writes are
  batched and propagated asynchronously.  Lower latency, but
  concurrent readers may temporarily disagree.
"""
