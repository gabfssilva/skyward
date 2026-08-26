from __future__ import annotations

from litestar import Controller, Request, delete, get, post, put
from litestar.openapi.datastructures import ResponseSpec
from litestar.params import Parameter
from litestar.response import Stream

from skyward.server.application import ports
from skyward.server.application.ssh import Result
from skyward.server.http.exceptions import failures
from skyward.shared.errors import CapabilityMismatchError

BYTES = "application/octet-stream"

NODE = "Which nodes to reach: `all`, or a rank."


def _target(node: str) -> ports.Target:
    """Read the ``node`` query as a target: every ready node, or the one at a rank."""
    if node == "all":
        return "all"
    return _rank(node)


async def _identified(computes: ports.Computes, ref: str) -> str:
    """The id behind a reference, because the links a compute is reached over are keyed by it.

    Every other route takes a name or an id and means the same compute by either.
    These reach the SSH connections this daemon is holding, which are indexed by
    id alone — so a name handed straight through would look like a compute nobody
    is connected to, which is a different answer entirely.
    """
    return (await computes.get(ref)).id


def _rank(node: str) -> int:
    """The same, for the endpoints that answer from exactly one node."""
    if not node.lstrip("-").isdigit():
        raise CapabilityMismatchError(f"node must be 'all' or a rank, not {node!r}")
    return int(node)


class FileController(Controller):
    path = "/computes/{compute_id:str}"
    tags = ["files"]

    @get(
        "/files",
        summary="List a path on the compute",
        description="`ls -la` of `path`, per node. Defaults to rank 0 — one listing is usually the question.",
        responses=failures(404, 409, 422),
    )
    async def ls(
        self,
        compute_id: str,
        files: ports.Files,
        computes: ports.Computes,
        path: str = Parameter(query="path", description="The path to list, on the node."),
        node: str = Parameter(query="node", default="0", description=NODE),
    ) -> dict[str, Result]:
        return dict(await files.ls(await _identified(computes, compute_id), _target(node), path))

    @delete(
        "/files",
        status_code=200,
        summary="Remove a path on the compute",
        description=(
            "`rm -rf` of `path`, per node. Defaults to every node, because a file left on one machine of four is "
            "the state a later broadcast reads and disagrees about."
        ),
        responses=failures(404, 409, 422),
    )
    async def rm(
        self,
        compute_id: str,
        files: ports.Files,
        computes: ports.Computes,
        path: str = Parameter(query="path", description="The path to remove, on the node."),
        node: str = Parameter(query="node", default="all", description=NODE),
    ) -> dict[str, Result]:
        return dict(await files.rm(await _identified(computes, compute_id), _target(node), path))

    @put(
        "/files",
        status_code=200,
        summary="Write a file onto the compute",
        description=(
            "The request body, written to `path`. Defaults to every node: code and data a task will read have to "
            "be wherever the task lands, and which node that is belongs to the dispatcher.\n\n"
            "The answer is per node — a machine that refused the write is a line of it, not the end of it."
        ),
        responses=failures(404, 409, 422),
    )
    async def upload(
        self,
        compute_id: str,
        request: Request,
        files: ports.Files,
        computes: ports.Computes,
        path: str = Parameter(query="path", description="Where to write it, on the node."),
        node: str = Parameter(query="node", default="all", description=NODE),
    ) -> dict[str, str | None]:
        return dict(await files.put(await _identified(computes, compute_id), _target(node), path, await request.body()))

    @get(
        "/files/content",
        media_type=BYTES,
        summary="Read a file off the compute",
        description=(
            "`path` from one node, as a raw byte stream. `node=all` is refused rather than picked between: four "
            "machines hold four files, and concatenating them would answer a question nobody asked."
        ),
        responses={
            200: ResponseSpec(
                bytes,
                media_type=BYTES,
                description="The file's bytes, as the node reads them off disk",
                generate_examples=False,
            ),
            **failures(404, 409, 422),
        },
    )
    async def download(
        self,
        compute_id: str,
        files: ports.Files,
        computes: ports.Computes,
        path: str = Parameter(query="path", description="The path to read, on the node."),
        node: str = Parameter(query="node", default="0", description="Which node to read from: a rank."),
    ) -> Stream:
        if node == "all":
            raise CapabilityMismatchError("a download comes from one node; name a rank")
        return Stream(files.get(await _identified(computes, compute_id), _rank(node), path), media_type=BYTES)

    @post(
        "/exec",
        status_code=200,
        summary="Run a shell command on the compute",
        description=(
            "`command` in a shell on each targeted node, and what each one said. The machine's shell, not the "
            "worker's — this reaches a node whose worker is busy, and answers questions about the machine rather "
            "than about the code running on it.\n\n"
            "A task is the other thing, and `POST /tasks` is where it goes."
        ),
        responses=failures(404, 409, 422),
    )
    async def run(
        self,
        compute_id: str,
        files: ports.Files,
        computes: ports.Computes,
        command: str = Parameter(query="command", description="The command line, run by the node's shell."),
        node: str = Parameter(query="node", default="all", description=NODE),
    ) -> dict[str, Result]:
        return dict(await files.run(await _identified(computes, compute_id), _target(node), command))
