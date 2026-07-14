from __future__ import annotations

from litestar import Request, Response

from skyward2.application.errors import SkywardError
from skyward2.protocol.schemas import Error


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
