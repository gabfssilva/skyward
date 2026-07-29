from __future__ import annotations

from typing import Any

import msgspec

from skyward.shared.schemas import Error


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


class UnexpectedResponseError(Exception):
    """A refusal the control plane did not write.

    A route that was never reached, a handler that crashed, a proxy in between —
    each answers with a body of its own, and none of them is an ``Error``.
    """

    def __init__(self, status: int, body: bytes) -> None:
        super().__init__(f"the control plane answered {status}: {body.decode(errors='replace')}")
        self.status = status
        self.body = body


def raised(error: Error) -> SkywardError:
    match error.code:
        case "task_failed":
            return TaskFailedError(error)
        case "task_indeterminate":
            return TaskIndeterminateError(error)
        case _:
            return SkywardError(error)


def refused(status: int, body: bytes) -> Exception:
    """The answer to a failed request, as the exception it deserves.

    Decoding a body that is not an ``Error`` as one raises a validation error about
    the shape of the answer, and that is what the caller is left holding instead of
    the answer — which is the only account of what went wrong there is.
    """
    try:
        return raised(msgspec.json.decode(body, type=Error))
    except msgspec.DecodeError:
        return UnexpectedResponseError(status, body)
