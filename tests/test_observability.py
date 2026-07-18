import io
import logging

import pytest

from skyward.observability import LogConfig, Logger, logger, setup_logging, teardown_logging
from skyward.observability.logger import NAME


@pytest.fixture(autouse=True)
def clean_sinks():
    logger.remove()
    logger.enable()
    yield
    logger.remove()
    logger.configure(patcher=None)


def sink(level="TRACE"):
    stream = io.StringIO()
    logger.add(stream, level=level)
    return stream


def test_bind_composes_fields():
    child = logger.bind(component="pool").bind(node_id="n-1")
    assert dict(child.fields) == {"component": "pool", "node_id": "n-1"}


def test_bind_does_not_mutate_parent():
    parent = logger.bind(component="pool")
    parent.bind(node_id="n-1")
    assert dict(parent.fields) == {"component": "pool"}
    assert dict(logger.fields) == {}


def test_bind_returns_a_new_logger():
    parent = logger.bind(component="pool")
    child = parent.bind(component="node")
    assert isinstance(child, Logger)
    assert child is not parent
    assert dict(child.fields) == {"component": "node"}


def test_message_formatting():
    stream = sink()
    logger.info("started {n} nodes", n=4)
    logger.info("started {} nodes", 4)
    lines = stream.getvalue().splitlines()
    assert all(line.endswith("started 4 nodes") for line in lines)
    assert len(lines) == 2


def test_levels_filter():
    stream = sink(level="WARNING")
    logger.trace("t")
    logger.debug("d")
    logger.info("i")
    logger.warning("w")
    logger.error("e")
    output = stream.getvalue()
    assert "w" in output
    assert "e" in output
    assert "| INFO" not in output
    assert "| DEBUG" not in output
    assert "| TRACE" not in output


def test_trace_level_is_emitted_below_debug():
    stream = sink()
    logger.trace("finest")
    assert "TRACE" in stream.getvalue()


def test_exception_attaches_traceback():
    stream = sink()
    try:
        raise ValueError("boom")
    except ValueError:
        logger.exception("failed")
    assert "ValueError: boom" in stream.getvalue()


def test_bound_fields_reach_the_record():
    seen: list[object] = []
    logger.configure(patcher=lambda record: seen.append(getattr(record, "extras", None)))
    sink()
    logger.bind(component="pool").info("hi")
    assert seen == [{"component": "pool"}]


def test_patcher_formats_context_into_the_line():
    stream = sink()
    ids = setup_logging(LogConfig(console=False, file=""))
    logger.add(stream, level="TRACE")
    logger.bind(component="pool", node_id="n-1").info("hi")
    teardown_logging(ids)
    assert "[component=pool node_id=n-1]" in stream.getvalue()


def test_remove_detaches_the_sink():
    stream = sink()
    handler_id = logger.add(io.StringIO(), level="TRACE")
    logger.remove(handler_id)
    logger.info("still here")
    assert "still here" in stream.getvalue()

    logger.remove()
    logger.info("gone")
    assert "gone" not in stream.getvalue()


def test_remove_unknown_id_is_a_noop():
    logger.remove(9999)


def test_disable_silences_and_enable_restores():
    stream = sink()
    logger.disable()
    logger.info("quiet")
    logger.enable()
    logger.info("loud")
    output = stream.getvalue()
    assert "quiet" not in output
    assert "loud" in output


def test_setup_logging_is_idempotent(tmp_path):
    config = LogConfig(file=str(tmp_path / "logs" / "sky.log"))
    first = setup_logging(config)
    second = setup_logging(config)
    assert len(first) == len(second) == 2
    assert len(logging.getLogger(NAME).handlers) == 2
    teardown_logging(second)


def test_teardown_logging_is_idempotent(tmp_path):
    ids = setup_logging(LogConfig(file=str(tmp_path / "sky.log")))
    teardown_logging(ids)
    teardown_logging(ids)
    assert logging.getLogger(NAME).handlers == []


def test_file_sink_writes(tmp_path):
    path = tmp_path / "nested" / "sky.log"
    ids = setup_logging(LogConfig(level="DEBUG", console=False, file=str(path)))
    logger.bind(component="pool").info("to disk")
    for handler in logging.getLogger(NAME).handlers:
        handler.flush()
    teardown_logging(ids)
    assert "to disk" in path.read_text()


def test_console_only_config_installs_one_sink():
    ids = setup_logging(LogConfig(console=True, file=""))
    assert len(ids) == 1
    teardown_logging(ids)


def test_the_logger_does_not_propagate():
    assert logging.getLogger(NAME).propagate is False


def test_the_daemons_loggers_still_reach_the_root(caplog):
    """The logger must not sit above ``skyward.*`` — the daemon logs there with the stdlib."""
    assert NAME.startswith("skyward.")
    with caplog.at_level(logging.ERROR, logger="skyward.server.exceptions"):
        logging.getLogger("skyward.server.exceptions").error("propagated")
    assert "propagated" in caplog.text
