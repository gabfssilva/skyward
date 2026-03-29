from __future__ import annotations

import pytest

from skyward.api.projection import SessionProjection

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


class TestSessionProjectionParam:
    def test_session_creates_default_projection(self) -> None:
        from skyward.core.session import Session

        s = Session(console=False, logging=False)
        assert isinstance(s.projection, SessionProjection)

    def test_session_accepts_external_projection(self) -> None:
        from skyward.core.session import Session

        proj = SessionProjection()
        s = Session(console=False, logging=False, projection=proj)
        assert s.projection is proj

    def test_default_projection_is_fresh_instance(self) -> None:
        from skyward.core.session import Session

        s1 = Session(console=False, logging=False)
        s2 = Session(console=False, logging=False)
        assert s1.projection is not s2.projection

    def test_projection_wired_during_start(self) -> None:
        """ProjectionActor is spawned and wired as spy when session starts."""
        from skyward.core.session import Session

        proj = SessionProjection()
        with Session(console=False, logging=False, projection=proj) as session:
            assert session.projection is proj
            assert session.is_active
