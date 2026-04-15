"""``Compute(server=True)`` is the default; ``daemon=`` was removed in D.

Workstream D.K2 removed the ``daemon=True`` shortcut from the
``Compute()`` context manager (the HTTP server subsumes that path).
Passing it now fails as an unexpected keyword argument.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


def test_daemon_kwarg_was_removed() -> None:
    import skyward as sky
    from skyward.core.compute import Compute

    with pytest.raises(TypeError, match="daemon"):
        with Compute(provider=sky.Container(), daemon=True) as _:  # type: ignore[call-arg]
            pass
