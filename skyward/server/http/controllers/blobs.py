from __future__ import annotations

from litestar import Controller, Request, Response, get, head, put

from skyward.server.application import ports

BLOB = "application/vnd.skyward.blob"


class BlobController(Controller):
    path = "/blobs"
    tags = ["blobs"]

    @head("/{sha256:str}", summary="Check existence")
    async def exists(self, sha256: str, blobs: ports.Blobs) -> None:
        await blobs.get(sha256)

    @put(
        "/{sha256:str}",
        status_code=201,
        summary="Upload a content-addressed blob",
        description=(
            "Small args go inline in `POST /tasks`. Above the threshold the SDK uploads here first and the task "
            "references the hash. Large results take the same path on the way back."
        ),
    )
    async def upload(self, sha256: str, request: Request, blobs: ports.Blobs) -> Response[None]:
        created = await blobs.put(sha256, await request.body())
        return Response(None, status_code=201 if created else 200)

    @get("/{sha256:str}", summary="Download a blob", media_type=BLOB)
    async def download(self, sha256: str, blobs: ports.Blobs) -> bytes:
        return await blobs.get(sha256)
