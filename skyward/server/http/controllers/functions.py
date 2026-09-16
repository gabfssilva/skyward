from __future__ import annotations

from litestar import Controller, Request, Response, get, head, post, put
from litestar.openapi.datastructures import ResponseSpec
from litestar.params import Parameter

from skyward.server.application import ports
from skyward.server.http.exceptions import failures
from skyward.shared import codec
from skyward.shared.schemas import Function, FunctionExcerpt, FunctionSource, Page
from skyward.worker.authored import authored

BLOB = "application/vnd.skyward.blob"


class FunctionController(Controller):
    path = "/functions"
    tags = ["functions"]

    @get(
        summary="List registered functions",
        description=(
            "Every upload this daemon has been handed, newest first. Uploading the same bytes twice adds no row, but "
            "one function is still many uploads: moving it down its file changes the pickle, and so does a run that "
            "captures a different value.\n\n"
            "So uploads are grouped. `lineage` is one function — the same name and qualname in the same file — and "
            "`version` counts the changes to its *code* along it, in order: an upload whose code matches the one "
            "before it is the same version, whatever else about the bytes moved. Reverting to older code is a new "
            "version, not the old number again, so the newest upload always carries the highest one.\n\n"
            "`latest` is one row per function, its newest upload. `lineage` is every upload of one function.\n\n"
            "The name, the file and the shape of the code are read off the payload without unpickling it. The text "
            "is not in the payload at all: `source` is a function written in the console, `excerpt` one the SDK "
            "uploaded along with what it uses from its module, and a function defined where there was no file to "
            "read has neither."
        ),
    )
    async def list(
        self,
        functions: ports.Functions,
        cursor: str | None = None,
        limit: int = Parameter(default=50, ge=1),
        latest: bool = Parameter(default=False, description="One row per function: its newest upload, which carries its highest version."),
        lineage: str | None = Parameter(default=None, description="Every upload of one function."),
    ) -> Page[Function]:
        return await functions.list(cursor, limit, latest, lineage)

    @post(
        status_code=201,
        summary="Write a function",
        description=(
            "A function as **text** instead of as a pickle, for a caller with no interpreter of its own — the browser "
            "console, or anything that speaks HTTP and nothing more.\n\n"
            "The body is a Python module and the name of the function in it to call. The daemon does not run any of "
            "it: the source is captured in a callable, that callable is pickled, and what is registered is the same "
            "blob `PUT` would have taken. The machine is still the only place a line of it executes, and it is "
            "compiled there once per call.\n\n"
            "Text that does not parse, or that defines no function under that name, is refused with "
            "`source_rejected` — both are wrong on every machine equally, and a dispatch is a slow place to find "
            "that out.\n\n"
            "The same text under the same name pickles to the same bytes, so writing it twice registers once. "
            "Editing it does not: a name takes in every version of its code, and each is a function of its own."
        ),
        responses={
            **failures(422),
            200: ResponseSpec(Function, description="Already registered — the same text, written again"),
        },
    )
    async def write(self, data: FunctionSource, functions: ports.Functions) -> Response[Function]:
        blob = await codec.payload.encode(authored(data.source, data.name))
        function, created = await functions.register(await codec.digest(blob), blob, data.name, data.source)
        return Response(function, status_code=201 if created else 200)

    @put(
        "/{sha256:str}/excerpt",
        status_code=200,
        summary="Attach a function's text",
        description=(
            "What the SDK sends after the pickle: the function as text, with the imports, constants, functions and "
            "classes of its module that it uses, in the order of the file. A pickle is compiled code and carries no "
            "text, so this is read where the function was defined and is the only way the console has anything to "
            "show for it.\n\n"
            "Stored as sent and never run. The function must already be registered."
        ),
        responses=failures(404),
    )
    async def excerpt(self, sha256: str, data: FunctionExcerpt, functions: ports.Functions) -> Function:
        return await functions.excerpt(sha256, data.text)

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
            "The size, the codec, the name it was registered under and, for one written as text, that text — never "
            "the pickle itself, which is a blob and is fetched as one."
        ),
        responses=failures(404),
    )
    async def read(self, sha256: str, functions: ports.Functions) -> Function:
        return await functions.get(sha256)
