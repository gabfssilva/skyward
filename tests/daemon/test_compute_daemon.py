from unittest.mock import MagicMock, patch

from skyward.accelerators import Accelerator
from skyward.api.spec import Spec
from skyward.core.compute import Compute
from skyward.daemon.fingerprint import compute_fingerprint
from skyward.daemon.pool import DaemonPool


class _FakeProvider:
    @property
    def type(self) -> str:
        return "aws"

    async def create_provider(self):  # noqa: ANN201
        raise NotImplementedError

    def default_options(self):  # noqa: ANN201
        return None


class TestComputeDaemon:
    @patch("skyward.daemon.spawn.ensure_daemon")
    def test_daemon_true_creates_daemon_pool(self, mock_ensure: MagicMock) -> None:
        original_enter = DaemonPool.__enter__
        original_exit = DaemonPool.__exit__
        DaemonPool.__enter__ = lambda self: self  # type: ignore[assignment]
        DaemonPool.__exit__ = lambda self, *a: None  # type: ignore[assignment]
        try:
            with Compute(
                provider=_FakeProvider(),
                accelerator=Accelerator("A100"),
                region="us-east-1",
                daemon=True,
            ) as pool:
                assert isinstance(pool, DaemonPool)
        finally:
            DaemonPool.__enter__ = original_enter  # type: ignore[assignment]
            DaemonPool.__exit__ = original_exit  # type: ignore[assignment]

    @patch("skyward.daemon.spawn.ensure_daemon")
    def test_daemon_pool_name_is_fingerprint(self, mock_ensure: MagicMock) -> None:
        original_enter = DaemonPool.__enter__
        original_exit = DaemonPool.__exit__
        DaemonPool.__enter__ = lambda self: self  # type: ignore[assignment]
        DaemonPool.__exit__ = lambda self, *a: None  # type: ignore[assignment]
        try:
            spec = Spec(provider=_FakeProvider(), accelerator=Accelerator("A100"), region="us-east-1")
            expected_name = compute_fingerprint(spec)
            with Compute(
                provider=_FakeProvider(),
                accelerator=Accelerator("A100"),
                region="us-east-1",
                daemon=True,
            ) as pool:
                assert pool._name == expected_name
        finally:
            DaemonPool.__enter__ = original_enter  # type: ignore[assignment]
            DaemonPool.__exit__ = original_exit  # type: ignore[assignment]

    @patch("skyward.daemon.spawn.ensure_daemon")
    def test_daemon_with_explicit_specs(self, mock_ensure: MagicMock) -> None:
        original_enter = DaemonPool.__enter__
        original_exit = DaemonPool.__exit__
        DaemonPool.__enter__ = lambda self: self  # type: ignore[assignment]
        DaemonPool.__exit__ = lambda self, *a: None  # type: ignore[assignment]
        try:
            with Compute(
                Spec(provider=_FakeProvider(), accelerator=Accelerator("A100"), region="us-east-1"),
                daemon=True,
            ) as pool:
                assert isinstance(pool, DaemonPool)
        finally:
            DaemonPool.__enter__ = original_enter  # type: ignore[assignment]
            DaemonPool.__exit__ = original_exit  # type: ignore[assignment]
