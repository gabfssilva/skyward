from __future__ import annotations

from typing import Any

from skyward2.protocol.schemas import Error


class SkywardError(Exception):
    """Something the control plane refused or could not do."""

    def __init__(self, error: Error) -> None:
        super().__init__(error.message)
        self.code = error.code
        self.message = error.message
        self.retryable = error.retryable
        self.details: dict[str, Any] = dict(error.details or {})


class TaskFailedError(SkywardError):
    """The user's function raised, and here is where."""

    def __str__(self) -> str:
        traceback = self.details.get("traceback")
        return f"{self.message}\n\n{traceback}" if traceback else self.message


class TaskIndeterminateError(SkywardError):
    """The call died and nobody knows whether the function ran.

    Never raised for a function that failed — only for one whose outcome was
    lost. Retrying is a decision about duplicate side effects, and it is the
    caller's to make.
    """


def raised(error: Error) -> SkywardError:
    match error.code:
        case "task_failed":
            return TaskFailedError(error)
        case "task_indeterminate":
            return TaskIndeterminateError(error)
        case _:
            return SkywardError(error)
