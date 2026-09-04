from __future__ import annotations

from litestar import Controller, Response, delete, get, patch, post, put
from litestar.openapi.datastructures import ResponseSpec
from litestar.params import Parameter

from skyward.server.application import ports
from skyward.server.application.reconciler import Wakeup
from skyward.server.http.exceptions import failures
from skyward.server.http.headers import etag, revision_of
from skyward.shared.schemas import (
    Compute,
    ComputeCreate,
    ComputeSpecPatch,
    ComputeState,
    Generation,
    GenerationCreate,
    Lease,
    LeaseClaim,
    Page,
)


class ComputeController(Controller):
    path = "/computes"
    tags = ["computes"]

    @get(
        summary="List computes",
        description=(
            "Every compute this daemon knows about, newest first, including the deleted ones — a compute's row "
            "outlives its machines. `live=true` lists only the computes that still owe something (`requested` through "
            "`deleting`), and `live=false` the finished ones. `owned=false` lists the orphans: computes no live "
            "process is holding a lease on, which is what a compute looks like between the script that made it "
            "exiting and the next one attaching."
        ),
        responses=failures(404),
    )
    async def list(
        self,
        computes: ports.Computes,
        cursor: str | None = None,
        limit: int = Parameter(default=50, ge=1),
        compute_state: ComputeState | None = Parameter(query="state", default=None),
        owned: bool | None = Parameter(default=None, description="`false` lists orphans — computes with no live owner."),
        live: bool | None = Parameter(default=None, description="`true` lists what is still running, `false` what is finished."),
    ) -> Page[Compute]:
        return await computes.list(cursor, limit, compute_state, owned, live)

    @post(
        status_code=201,
        summary="Create a compute",
        description=(
            "Idempotent by `Idempotency-Key`: the same key with the same payload returns the original resource with "
            "`200`; with a different payload, `409 idempotency_conflict`.\n\n"
            "Does not wait for readiness — returns `status.state: requested` and the client observes the resource or the "
            "event stream. There is no `operation` resource: `generation` vs `status.observed_generation` is the "
            "progress."
        ),
        responses={**failures(409, 422), 200: ResponseSpec(Compute, description="The compute this key already created")},
    )
    async def create(
        self,
        data: ComputeCreate,
        computes: ports.Computes,
        wake: Wakeup,
        idempotency_key: str = Parameter(header="Idempotency-Key"),
    ) -> Response[Compute]:
        compute, created = await computes.create(data, idempotency_key)
        wake("compute.changed", compute_id=compute.id)
        return Response(compute, status_code=201 if created else 200, headers=etag(compute.revision))

    @get(
        "/{compute_id:str}",
        summary="Read a compute",
        description="Accepts an id or a name. The response always carries both.",
        responses=failures(404),
    )
    async def read(self, compute_id: str, computes: ports.Computes) -> Response[Compute]:
        compute = await computes.get(compute_id)
        return Response(compute, headers=etag(compute.revision))

    @patch(
        "/{compute_id:str}",
        summary="Change a compute's spec",
        description=(
            "Only `spec.nodes` is mutable in place: it bumps `generation` and the reconciler resizes with drain.\n\n"
            "A compute running a collective plugin (`torch`, `jax`, `accelerate`) is refused with `422 "
            "compute_not_resizable`: its process group is formed on the first task and never formed again, so a rank "
            "added afterwards blocks in it.\n\n"
            "Changing immutable fields (provider, image, worker, plugins, volumes, ports) is **drift**: it is recorded in "
            "`status.drift`, the applied definition is kept, and no infrastructure is replaced. Replacing requires "
            "`POST /computes/{id}/generations`."
        ),
        responses=failures(404, 412, 422),
    )
    async def update(
        self,
        compute_id: str,
        data: ComputeSpecPatch,
        computes: ports.Computes,
        wake: Wakeup,
        if_match: str = Parameter(header="If-Match"),
    ) -> Response[Compute]:
        compute = await computes.patch(compute_id, data, revision_of(if_match))
        wake("compute.changed", compute_id=compute.id)
        return Response(compute, headers=etag(compute.revision))

    @delete(
        "/{compute_id:str}",
        status_code=202,
        summary="Mark a compute for destruction",
        description=(
            "Writes `spec.desired: deleted`. Not synchronous, and not a detach: reconciliation continues until the "
            "provider **confirms the resources are gone**, and only then does `status.state` become `deleted`.\n\n"
            "No process shutdown ever issues this command."
        ),
        responses=failures(404, 409, 412),
    )
    async def destroy(
        self,
        compute_id: str,
        computes: ports.Computes,
        wake: Wakeup,
        if_match: str = Parameter(header="If-Match"),
        idempotency_key: str = Parameter(header="Idempotency-Key"),
    ) -> Response[Compute]:
        compute = await computes.delete(compute_id, revision_of(if_match), idempotency_key)
        wake("compute.changed", compute_id=compute.id)
        return Response(compute, status_code=202, headers=etag(compute.revision))

    @get(
        "/{compute_id:str}/generations",
        summary="List definition history",
        description="Every definition this compute has had, newest last. A rollback is a generation too, so this grows.",
        responses=failures(404),
    )
    async def list_generations(self, compute_id: str, generations: ports.Generations) -> Page[Generation]:
        return await generations.list(compute_id)

    @get(
        "/{compute_id:str}/generations/{number:int}",
        summary="Read a generation",
        description="One definition as it was frozen, and whether the machines were ever built to match it.",
        responses=failures(404),
    )
    async def get_generation(self, compute_id: str, number: int, generations: ports.Generations) -> Generation:
        return await generations.get(compute_id, number)

    @post(
        "/{compute_id:str}/generations",
        status_code=202,
        summary="Create a generation (apply drift, or roll back)",
        description=(
            "Replacing infrastructure *is* creating a new generation: quiesce, drain, destroy the old one, provision the "
            "new one — same `compute_id`.\n\n"
            "Without `source`, applies the drift pending in `status.drift`. With `source`, rolls back to that "
            "generation.\n\n"
            "`force: true` marks unresolved tasks as `indeterminate` before replacing; `force: false` refuses while "
            "executions are still active."
        ),
        responses=failures(404, 409, 412, 422),
    )
    async def create_generation(
        self,
        compute_id: str,
        data: GenerationCreate,
        generations: ports.Generations,
        wake: Wakeup,
        if_match: str = Parameter(header="If-Match"),
        idempotency_key: str = Parameter(header="Idempotency-Key"),
    ) -> Generation:
        generation = await generations.create(compute_id, data, revision_of(if_match), idempotency_key)
        wake("compute.changed", compute_id=compute_id)
        return generation

    @put(
        "/{compute_id:str}/lease",
        summary="Claim or renew ownership",
        description=(
            "A compute has at most one live owner per generation; zero is legitimate and temporary (daemon restarting, "
            "script killed).\n\n"
            "Claiming requires the current lease to be expired or absent — it is a CAS. Renewing requires being the "
            "current owner. Losing renewals destroys nothing by itself: if `spec.delete_on_exit` is `true`, "
            "reconciliation tears the compute down; if `false`, it simply sits ownerless until something adopts it."
        ),
        responses=failures(404, 409),
    )
    async def claim_lease(self, compute_id: str, data: LeaseClaim, computes: ports.Computes) -> Lease:
        return await computes.claim_lease(compute_id, data)

    @delete(
        "/{compute_id:str}/lease",
        status_code=204,
        summary="Release ownership",
        description="Orderly detach: drops the claim without touching `spec.desired`. Destroys nothing.",
        responses=failures(404),
    )
    async def release_lease(self, compute_id: str, computes: ports.Computes) -> None:
        await computes.release_lease(compute_id)
