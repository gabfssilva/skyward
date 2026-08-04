from __future__ import annotations

from litestar import Controller, Request, Response, get, head, put
from litestar.openapi.datastructures import ResponseSpec

from skyward.server.application import ports
from skyward.server.http.exceptions import failures

BLOB = "application/vnd.skyward.blob"


class BlobController(Controller):
    path = "/blobs"
    tags = ["blobs"]

    @head(
        "/{sha256:str}",
        summary="Check existence",
        description="Whether this content is already here, so a client can skip an upload it does not need to make.",
        responses=failures(404),
    )
    async def exists(self, sha256: str, blobs: ports.Blobs) -> None:
        await blobs.get(sha256)

    @put(
        "/{sha256:str}",
        status_code=201,
        summary="Upload a content-addressed blob",
        description=(
            "Small args go inline in `POST /tasks`. Above the threshold the SDK uploads here first and the task "
            "references the hash. Large results take the same path on the way back.\n\n"
            "The body is the bytes themselves, and the name is what they hash to — content that does not hash to its "
            "name is refused with `hash_mismatch` rather than stored under a name that lies. Uploading content already "
            "here answers `200` and writes nothing."
        ),
        responses={**failures(400), 200: ResponseSpec(None, description="Already stored — the upload was a no-op")},
    )
    async def upload(self, sha256: str, request: Request, blobs: ports.Blobs) -> Response[None]:
        created = await blobs.put(sha256, await request.body())
        return Response(None, status_code=201 if created else 200)

    @get(
        "/{sha256:str}",
        summary="Download a blob",
        media_type=BLOB,
        description=(
            "Reading does not consume it. A result read twice is the same bytes twice, which is what lets an SDK that "
            "restarted collect a result the process before it had already taken."
        ),
        responses={
            **failures(404),
            200: ResponseSpec(bytes, media_type=BLOB, description="The stored bytes", generate_examples=False),
        },
    )
    async def download(self, sha256: str, blobs: ports.Blobs) -> bytes:
        return await blobs.get(sha256)
