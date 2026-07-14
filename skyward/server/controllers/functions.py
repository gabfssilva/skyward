from __future__ import annotations

from litestar import Controller, Request, Response, get, head, put
from litestar.params import Parameter

from skyward.application import ports
from skyward.protocol.schemas import Function, Page

BLOB = "application/vnd.skyward.blob"


class FunctionController(Controller):
    path = "/functions"
    tags = ["functions"]

    @get(summary="List registered functions")
    async def list(self, functions: ports.Functions, cursor: str | None = None, limit: int = 50) -> Page[Function]:
        return await functions.list(cursor, limit)

    @head(
        "/{sha256:str}",
        summary="Check whether a function is already registered",
        description=(
            "The SDK calls this before uploading the blob. A function is uploaded **once**, no matter how many tasks "
            "call it — content-addressing is what makes `function` a cheap resource instead of a pickle repeated on "
            "every dispatch."
        ),
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

    @get("/{sha256:str}", summary="Read a function's metadata")
    async def read(self, sha256: str, functions: ports.Functions) -> Function:
        return await functions.get(sha256)
