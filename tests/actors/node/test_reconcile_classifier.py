"""Unit tests for ``_classify_reconcile_result``.

Maps a worker ``GetResult`` reply (or a network failure) to one of:
``"succeeded" | "failed" | "wait" | "interrupted"``. Caller decides what
to surface to the task manager based on the decision.
"""
from __future__ import annotations

import pytest

from skyward.actors.node.actor import _classify_reconcile_result
from skyward.infra.worker import (
    ResultDone,
    ResultPending,
    ResultUnknown,
    TaskFailed,
    TaskSucceeded,
)

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("unit")]


class TestClassifyReconcileResult:
    def test_pending_means_wait(self) -> None:
        assert _classify_reconcile_result(ResultPending(), None) == "wait"

    def test_done_with_success_payload(self) -> None:
        reply = ResultDone(result=TaskSucceeded(result=42, node_id=0))
        assert _classify_reconcile_result(reply, None) == "succeeded"

    def test_done_with_failure_payload(self) -> None:
        reply = ResultDone(result=TaskFailed(
            error="boom", traceback="...", node_id=0,
        ))
        assert _classify_reconcile_result(reply, None) == "failed"

    def test_unknown_means_interrupted(self) -> None:
        assert _classify_reconcile_result(ResultUnknown(), None) == "interrupted"

    def test_none_reply_means_interrupted(self) -> None:
        """Network failure / ask raised → no reply → interrupted."""
        assert _classify_reconcile_result(None, "ask timed out") == "interrupted"

    def test_unexpected_reply_means_interrupted(self) -> None:
        """Defensive: an unmatched reply type is treated as fatal."""
        assert _classify_reconcile_result(object(), None) == "interrupted"
