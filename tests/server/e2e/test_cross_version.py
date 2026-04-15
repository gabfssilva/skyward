"""J2: cross-version client/server matrix over Docker.

The matrix below mirrors the plan (§J2): six directed pairs across
``{3.12, 3.13, 3.14}``. Each pair brings up a server container on one
Python version and a client container on another, runs a trivial
``sky.function`` through :class:`ServerPool`, and asserts the result.

These tests are slow and require a Docker daemon; they are skipped by
default (``@pytest.mark.e2e``) and only run when ``task test:e2e``
enables the marker.
"""
from __future__ import annotations

import shutil

import pytest

pytestmark = [pytest.mark.e2e]


_DOCKER_AVAILABLE = shutil.which("docker") is not None


_PAIRS: list[tuple[str, str]] = [
    ("3.12", "3.13"),
    ("3.13", "3.12"),
    ("3.12", "3.14"),
    ("3.14", "3.12"),
    ("3.13", "3.14"),
    ("3.14", "3.13"),
]


@pytest.mark.parametrize(("server_py", "client_py"), _PAIRS)
@pytest.mark.skipif(not _DOCKER_AVAILABLE, reason="docker not available")
def test_cross_version_round_trip(server_py: str, client_py: str) -> None:
    """Full-stack round trip across two mismatched interpreters.

    Skipped until the container images + fixture runner land — see
    the design plan (§J2) for the concrete Docker harness. The test
    body is intentionally a placeholder so the matrix is visible.
    """
    pytest.skip(
        f"harness not wired yet (server={server_py}, client={client_py})",
    )
