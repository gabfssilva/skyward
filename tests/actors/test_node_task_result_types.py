"""Verify the TaskResult union subtypes."""

from skyward.actors.messages import TaskFailed, TaskInterrupted, TaskSucceeded


class TestTaskResultSubtypes:
    def test_pattern_matching(self):
        results = [
            TaskSucceeded(value=b"ok", node_id=1, task_id="t1"),
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
