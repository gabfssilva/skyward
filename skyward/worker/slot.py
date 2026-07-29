"""This process's index among the worker's concurrent slots.

Zero in the worker itself and under a thread executor, where every task shares the
one process. Distinct per child under a reused process executor, set once when the
child is initialised — the datum a plugin uses to pin each concurrent task to its
own share of the machine.
"""

from __future__ import annotations

_slot = 0


def get() -> int:
    return _slot


def set(index: int) -> None:
    global _slot
    _slot = index
