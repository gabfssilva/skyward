from __future__ import annotations

import logging

from litestar import Request, Response
from litestar.exceptions import HTTPException
from litestar.exceptions.responses import create_exception_response

from skyward.application.errors import SkywardError
from skyward.protocol.schemas import Error

logger = logging.getLogger(__name__)


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
            logger.error("unhandled error: %s %s", request.method, request.url.path, exc_info=exc)
    return create_exception_response(request, exc)
