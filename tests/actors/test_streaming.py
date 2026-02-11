from skyward.actors.messages import (
    BootstrapCommand,
    BootstrapConsole,
    BootstrapFailed,
    BootstrapPhase,
    InstanceMetadata,
    Log,
    Metric,
    StopMonitor,
    _StreamedEvent,
    _StreamEnded,
)
from skyward.actors.streaming import _convert, instance_monitor


def _test_instance() -> InstanceMetadata:
    return InstanceMetadata(
        id="i-test",
        node=0,
        provider="aws",
        ip="1.2.3.4",
        ssh_port=22,
    )


class TestConvertConsole:
    def test_stdout(self):
        from skyward.infra.ssh import RawBootstrapConsole

        info = _test_instance()
        event = _convert(RawBootstrapConsole(content="installing deps...", stream="stdout"), info)
        assert isinstance(event, BootstrapConsole)
        assert event.content == "installing deps..."
        assert event.stream == "stdout"
        assert event.instance is info

    def test_stderr(self):
        from skyward.infra.ssh import RawBootstrapConsole

        info = _test_instance()
        event = _convert(RawBootstrapConsole(content="warning: deprecated", stream="stderr"), info)
        assert isinstance(event, BootstrapConsole)
        assert event.stream == "stderr"


class TestConvertPhase:
    def test_started(self):
        from skyward.infra.ssh import RawBootstrapPhase

        info = _test_instance()
        event = _convert(RawBootstrapPhase(event="started", phase="apt"), info)
        assert isinstance(event, BootstrapPhase)
        assert event.event == "started"
        assert event.phase == "apt"

    def test_completed_with_elapsed(self):
        from skyward.infra.ssh import RawBootstrapPhase

        info = _test_instance()
        event = _convert(RawBootstrapPhase(event="completed", phase="pip", elapsed=12.5), info)
        assert isinstance(event, BootstrapPhase)
        assert event.event == "completed"
        assert event.elapsed == 12.5

    def test_failed_returns_bootstrap_failed(self):
        from skyward.infra.ssh import RawBootstrapPhase

        info = _test_instance()
        event = _convert(RawBootstrapPhase(event="failed", phase="uv", error="timeout"), info)
        assert isinstance(event, BootstrapFailed)
        assert event.phase == "uv"
        assert event.error == "timeout"

    def test_failed_defaults_error(self):
        from skyward.infra.ssh import RawBootstrapPhase

        info = _test_instance()
        event = _convert(RawBootstrapPhase(event="failed", phase="setup"), info)
        assert isinstance(event, BootstrapFailed)
        assert event.error == "unknown"


class TestConvertCommand:
    def test_command(self):
        from skyward.infra.ssh import RawBootstrapCommand

        info = _test_instance()
        event = _convert(RawBootstrapCommand(command="pip install torch"), info)
        assert isinstance(event, BootstrapCommand)
        assert event.command == "pip install torch"

    def test_empty_command(self):
        from skyward.infra.ssh import RawBootstrapCommand

        info = _test_instance()
        event = _convert(RawBootstrapCommand(command=""), info)
        assert isinstance(event, BootstrapCommand)
        assert event.command == ""


class TestConvertMetric:
    def test_metric(self):
        from skyward.infra.ssh import RawMetricEvent

        info = _test_instance()
        event = _convert(RawMetricEvent(name="gpu_util", value=95.5, ts=1700000000.0), info)
        assert isinstance(event, Metric)
        assert event.name == "gpu_util"
        assert event.value == 95.5
        assert event.timestamp == 1700000000.0

    def test_metric_invalid_value_returns_none(self):
        from skyward.infra.ssh import RawMetricEvent

        info = _test_instance()
        assert _convert(RawMetricEvent(name="gpu_util", value="not-a-number", ts=0.0), info) is None


class TestConvertLog:
    def test_log_stdout(self):
        from skyward.infra.ssh import RawLogEvent

        info = _test_instance()
        event = _convert(RawLogEvent(content="training epoch 1", stream="stdout"), info)
        assert isinstance(event, Log)
        assert event.line == "training epoch 1"
        assert event.stream == "stdout"

    def test_log_stderr(self):
        from skyward.infra.ssh import RawLogEvent

        info = _test_instance()
        event = _convert(RawLogEvent(content="error occurred", stream="stderr"), info)
        assert isinstance(event, Log)
        assert event.stream == "stderr"


class TestConvertEdgeCases:
    def test_unknown_type_returns_none(self):
        info = _test_instance()
        assert _convert("unknown-object", info) is None

    def test_none_returns_none(self):
        info = _test_instance()
        assert _convert(None, info) is None


class TestMessages:
    def test_stop_monitor(self):
        msg = StopMonitor()
        assert isinstance(msg, StopMonitor)

    def test_stream_ended(self):
        msg = _StreamEnded()
        assert msg.error is None

    def test_stream_ended_with_error(self):
        msg = _StreamEnded(error="connection lost")
        assert msg.error == "connection lost"

    def test_instance_monitor_callable(self):
        assert callable(instance_monitor)
