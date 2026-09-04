from __future__ import annotations

from litestar import Controller, Request, Response, get, head, put
from litestar.openapi.datastructures import ResponseSpec
from litestar.params import Parameter

from skyward.server.application import ports
from skyward.server.http.exceptions import failures
from skyward.shared.schemas import Function, Page

BLOB = "application/vnd.skyward.blob"


class FunctionController(Controller):
    path = "/functions"
    tags = ["functions"]

    @get(
        summary="List registered functions",
        description="Every function this daemon has been handed, by hash. Uploading the same code twice adds no row.",
    )
    async def list(self, functions: ports.Functions, cursor: str | None = None, limit: int = Parameter(default=50, ge=1)) -> Page[Function]:
        return await functions.list(cursor, limit)

    @head(
        "/{sha256:str}",
        summary="Check whether a function is already registered",
        description=(
            "The SDK calls this before uploading the blob. A function is uploaded **once**, no matter how many tasks "
            "call it — content-addressing is what makes `function` a cheap resource instead of a pickle repeated on "
            "every dispatch."
        ),
        responses=failures(404),
    )
    async def exists(self, sha256: str, functions: ports.Functions) -> None:
        await functions.get(sha256)

    @put(
        "/{sha256:str}",
        status_code=201,
        summary="Register a function",
        description=(
            "The body is the binary envelope (cloudpickle + compression). The server recomputes the hash over the "
            "serialized bytes before compression and rejects with `hash_mismatch` if it disagrees with the path.\n\n"
            "Accepting cloudpickle is accepting arbitrary code execution. In local single-user mode that is explicit: "
            "the caller is the user themselves."
        ),
        responses={
            **failures(400),
            200: ResponseSpec(Function, description="Already registered — the upload was a no-op"),
        },
    )
    async def register(
        self,
        sha256: str,
        request: Request,
        functions: ports.Functions,
        name: str | None = Parameter(header="X-Skyward-Function-Name", default=None),
    ) -> Response[Function]:
        function, created = await functions.register(sha256, await request.body(), name)
        return Response(function, status_code=201 if created else 200)

    @get(
        "/{sha256:str}",
        summary="Read a function's metadata",
        description=(
            "The size, the codec and the name it was registered under — never the code itself, which is a blob and is "
            "fetched as one."
        ),
        responses=failures(404),
    )
    async def read(self, sha256: str, functions: ports.Functions) -> Function:
        return await functions.get(sha256)
