"""Verify the TaskResult union subtypes."""

from skyward.actors.messages import TaskFailed, TaskInterrupted, TaskSucceeded


class TestTaskResultSubtypes:
    def test_succeeded_has_value_and_node_id(self):
        r = TaskSucceeded(value=42, node_id=1, task_id="t1")
        assert r.value == 42
        assert r.node_id == 1
        assert r.task_id == "t1"

    def test_failed_has_error_and_node_id(self):
        err = ValueError("bad input")
        r = TaskFailed(error=err, node_id=1, task_id="t1")
        assert r.error is err
        assert r.node_id == 1

    def test_interrupted_has_error_and_node_id(self):
        err = RuntimeError("Node 3 child stopped")
        r = TaskInterrupted(error=err, node_id=3, task_id="t2")
        assert r.error is err
        assert r.node_id == 3

    def test_pattern_matching(self):
        results = [
            TaskSucceeded(value="ok", node_id=1, task_id="t1"),
            TaskFailed(error=ValueError("x"), node_id=1, task_id="t2"),
            TaskInterrupted(error=RuntimeError("lost"), node_id=1, task_id="t3"),
        ]
        kinds = []
        for r in results:
            match r:
                case TaskSucceeded():
                    kinds.append("success")
                case TaskFailed():
                    kinds.append("user_error")
                case TaskInterrupted():
                    kinds.append("infra_error")
        assert kinds == ["success", "user_error", "infra_error"]
