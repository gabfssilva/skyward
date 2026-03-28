import pytest
from unittest.mock import MagicMock

from skyward.actors.pool.messages import (
    PoolMsg,
    PoolStarted,
    ProvisionFailed,
    RecoverPool,
)


class TestPoolRecovery:
    @pytest.mark.asyncio
    async def test_recover_pool_message_is_frozen(self) -> None:
        msg = RecoverPool(
            spec=MagicMock(), provider=MagicMock(),
            cluster=MagicMock(), instances=(),
            reply_to=MagicMock(),
        )
        with pytest.raises(AttributeError):
            msg.spec = MagicMock()  # type: ignore[misc]
