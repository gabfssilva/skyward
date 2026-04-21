from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from skyward.actors.messages import NodeJoined
from skyward.actors.pool.actor import _notify_scaling
from skyward.actors.pool.state import PoolState


@pytest.mark.unit
def test_notify_scaling_broadcasts_to_both_refs() -> None:
    rec, autos = MagicMock(), MagicMock()
    s = PoolState(
        spec=MagicMock(), provider=MagicMock(), reply_to=MagicMock(),
        reconciler_ref=rec, autoscaler_ref=autos,
    )
    msg = NodeJoined(node_id=1)
    _notify_scaling(s, msg)
    rec.tell.assert_called_once_with(msg)
    autos.tell.assert_called_once_with(msg)
