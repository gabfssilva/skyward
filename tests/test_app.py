"""The daemon as a guest: building it must not reach into the host's logging."""

from __future__ import annotations

import logging

import pytest

from skyward.server.app import create_app, mock_services

pytestmark = pytest.mark.unit


def test_building_the_embedded_app_leaves_the_root_logger_alone():
    root = logging.getLogger()
    level, handlers = root.level, root.handlers[:]
    create_app(mock_services())
    assert root.level == level
    assert root.handlers == handlers


def test_a_standalone_daemon_may_opt_into_logging():
    root = logging.getLogger()
    level, handlers = root.level, root.handlers[:]
    try:
        create_app(mock_services(), logging=True)
        assert root.level == logging.INFO
    finally:
        root.setLevel(level)
        root.handlers = handlers
