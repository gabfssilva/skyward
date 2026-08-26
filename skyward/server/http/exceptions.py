from __future__ import annotations

from litestar import Request, Response
from litestar.exceptions import HTTPException
from litestar.exceptions.responses import create_exception_response
from litestar.openapi.datastructures import ResponseSpec

from skyward.shared.errors import SkywardError
from skyward.shared.observability import logger
from skyward.shared.schemas import Error

logger = logger.bind(component="http")

FAILURES: dict[int, str] = {
    400: "Content does not hash to the name it was given — `hash_mismatch`",
    404: "No such resource — `not_found`",
    409: (
        "The write conflicts with what is already here — `idempotency_conflict`, `lease_held`, `name_taken`, "
        "`compute_not_connected`, `task_failed`, `task_indeterminate`, `duplication_not_acknowledged`"
    ),
    412: "`If-Match` did not name the stored revision — `revision_conflict`. Retryable: re-read, re-apply, re-send",
    422: (
        "Well-formed and unsatisfiable — `compute_not_accepting`, `compute_not_resizable`, `capability_mismatch`, "
        "`unsupported_provider`, `unsupported_plugin`, `secret_in_definition`"
    ),
}
"""What each status means when the body is an :class:`Error`.

The codes are the closed set a client matches on; the status is the coarse
answer HTTP has for the same thing. Both are given because a caller that only
reads one of them is a caller that has to guess.
"""


def failures(*statuses: int) -> dict[int, ResponseSpec]:
    """The failures a route can answer with, in the one shape all of them take.

    Declared per route rather than applied to every route at once: which of these
    a route can actually produce is a fact about the route, and a spec that
    promised all five everywhere would be documenting the exception handler
    instead of the endpoint.
    """
    return {status: ResponseSpec(Error, description=FAILURES[status], generate_examples=False) for status in statuses}


def skyward_error_handler(request: Request, exc: SkywardError) -> Response[Error]:
    return Response(
        Error(
            code=exc.code,
            message=exc.message,
            retryable=exc.retryable,
            request_id=request.headers.get("X-Request-ID"),
            details=exc.details or None,
        ),
        status_code=exc.status,
    )


def unhandled_error_handler(request: Request, exc: Exception) -> Response:
    """Whatever the control plane did not mean to do, said out loud.

    Embedded, the daemon has no logging config of its own, so an exception no
    handler claimed becomes a bare 500 and its traceback goes nowhere — the caller
    is told the control plane broke and nothing about where. The answer is still
    the one Litestar would have written; what is new is the record of what it was for.
    """
    match exc:
        case HTTPException(status_code=status) if status < 500:
            pass
        case _:
            logger.error("unhandled error: {} {}", request.method, request.url.path, exc_info=exc)
    return create_exception_response(request, exc)
