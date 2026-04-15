"""UUIDv7 correctness and monotonicity."""
from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


def test_uuid7_version_bits() -> None:
    from skyward.infra.ids import uuid7

    u = uuid7()
    assert u.version == 7


def test_uuid7_variant_bits() -> None:
    from skyward.infra.ids import uuid7

    u = uuid7()
    assert u.variant == "specified in RFC 4122"


def test_uuid7_monotonic_within_tight_loop() -> None:
    from skyward.infra.ids import uuid7

    last = uuid7()
    for _ in range(1000):
        current = uuid7()
        assert current > last
        last = current


def test_uuid7_distinct_across_threads() -> None:
    import threading

    from skyward.infra.ids import uuid7

    seen: set[str] = set()
    lock = threading.Lock()

    def worker() -> None:
        batch = {str(uuid7()) for _ in range(100)}
        with lock:
            seen.update(batch)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(seen) == 8 * 100
