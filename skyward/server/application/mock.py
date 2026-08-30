from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime

from skyward.shared.errors import NotFoundError
from skyward.shared.schemas import (
    Compute,
    ComputeCreate,
    ComputeSpec,
    ComputeSpecPatch,
    ComputeState,
    ComputeStatus,
    DependencyState,
    Execution,
    ExecutionCreate,
    Function,
    Generation,
    GenerationCreate,
    Image,
    Lease,
    LeaseClaim,
    Node,
    NodeBounds,
    Offer,
    Page,
    Provider,
    ProviderCreate,
    ProviderRef,
    RetryPolicy,
    Spec,
    Task,
    TaskCreate,
    TaskState,
    Worker,
)

NOW = datetime(2026, 7, 13, 12, 0, 0, tzinfo=UTC)

SPEC = ComputeSpec(
    specs=(Spec(provider=ProviderRef(kind="aws"), accelerator="a100", accelerator_count=1, region="us-east-1"),),
    nodes=NodeBounds(initial=4, min=2, max=8),
    image=Image(python="3.13", pip=("torch",)),
    worker=Worker(concurrency=8, executor="process"),
    delete_on_exit=True,
)

COMPUTE = Compute(
    id="cmp_7f3a1c",
    name="training",
    revision=7,
    generation=3,
    spec=SPEC,
    status=ComputeStatus(state="ready", observed_generation=3, nodes_ready=4, nodes_total=4),
    lease=Lease(owner="ctl_1:epoch_9", expires_at=NOW),
    created_at=NOW,
)

NODE = Node(
    id="nod_c19e40",
    compute_id=COMPUTE.id,
    generation=3,
    rank=0,
    revision=2,
    desired="present",
    state="ready",
    provider_binding={"instance_id": "i-0abc123", "region": "us-east-1", "volume_ids": ["vol-01"]},
    created_at=NOW,
    address="10.0.1.7",
    accelerator="a100",
    price_per_hour=1.42,
)

DEAD_NODE = Node(
    id="nod_5b2d81",
    compute_id=COMPUTE.id,
    generation=3,
    rank=1,
    revision=5,
    desired="deleted",
    state="lost",
    provider_binding={"instance_id": "i-0def456", "region": "us-east-1", "volume_ids": ["vol-02"]},
    created_at=NOW,
    terminated_at=NOW,
)

FUNCTION = Function(
    sha256="a" * 64,
    size_bytes=18_422,
    codec="cloudpickle+lz4",
    created_at=NOW,
    name="train",
)

EXECUTION = Execution(
    id="exe_44b0aa",
    rank=0,
    ordinal=1,
    state="succeeded",
    node_id=NODE.id,
    result_sha256="b" * 64,
    started_at=NOW,
    finished_at=NOW,
)

TASK = Task(
    id="tsk_9d21f0",
    compute_id=COMPUTE.id,
    generation=3,
    function=FUNCTION.sha256,
    args_sha256="c" * 64,
    dispatch="one",
    state="succeeded",
    retry=RetryPolicy(),
    executions=(EXECUTION,),
    submitted_at=NOW,
    result_sha256=EXECUTION.result_sha256,
    finished_at=NOW,
)

GENERATION = Generation(number=3, spec=SPEC, hash="d" * 64, created_at=NOW, applied=True)

PROVIDER = Provider(
    id="prv_0a1b2c3d4e5f",
    name="vast-main",
    kind="vastai",
    config={"min_reliability": 0.95},
    offers_ttl_seconds=120,
    created_at=NOW,
    offers_fetched_at=NOW,
    offers_count=412,
)

OFFER = Offer(
    id="12345678",
    provider_id=PROVIDER.id,
    provider_name=PROVIDER.name,
    kind=PROVIDER.kind,
    instance_type="RTX 4090",
    accelerator="4090",
    accelerator_count=2,
    cpus=16,
    memory_gb=64.0,
    disk_gb=500.0,
    region="US-CA",
    spot_price=0.31,
    on_demand_price=0.62,
    available=2,
    fetched_at=NOW,
    expires_at=NOW,
    specific={"machine_id": 9001, "reliability": 0.987},
)


class MockComputes:
    async def create(self, body: ComputeCreate, idempotency_key: str) -> tuple[Compute, bool]:
        return COMPUTE, True

    async def get(self, ref: str) -> Compute:
        return COMPUTE

    async def list(self, cursor: str | None, limit: int, state: ComputeState | None, owned: bool | None, live: bool | None) -> Page[Compute]:
        return Page(items=(COMPUTE,))

    async def patch(self, ref: str, body: ComputeSpecPatch, expected_revision: int) -> Compute:
        return COMPUTE

    async def delete(self, ref: str, expected_revision: int, idempotency_key: str) -> Compute:
        return COMPUTE

    async def claim_lease(self, ref: str, claim: LeaseClaim) -> Lease:
        return Lease(owner=claim.owner, expires_at=NOW)

    async def release_lease(self, ref: str) -> None:
        return None


class MockGenerations:
    async def list(self, compute: str) -> Page[Generation]:
        return Page(items=(GENERATION,))

    async def get(self, compute: str, number: int) -> Generation:
        return GENERATION

    async def create(self, compute: str, body: GenerationCreate, expected_revision: int, idempotency_key: str) -> Generation:
        return GENERATION


class MockNodes:
    async def list(self, compute: str, include_terminal: bool, generation: int | None) -> Page[Node]:
        return Page(items=(NODE, DEAD_NODE) if include_terminal else (NODE,))

    async def get(self, compute: str, node_id: str) -> Node:
        return NODE

    async def drain(self, compute: str, node_id: str, idempotency_key: str) -> Node:
        return DEAD_NODE


class MockFunctions:
    async def exists(self, sha256: str) -> bool:
        return True

    async def get(self, sha256: str) -> Function:
        return FUNCTION

    async def list(self, cursor: str | None, limit: int) -> Page[Function]:
        return Page(items=(FUNCTION,))

    async def register(self, sha256: str, blob: bytes, name: str | None) -> tuple[Function, bool]:
        return FUNCTION, True


class MockBlobs:
    async def exists(self, sha256: str) -> bool:
        return True

    async def put(self, sha256: str, blob: bytes) -> bool:
        return True

    async def get(self, sha256: str) -> bytes:
        return b"\x00mock-blob"


class MockTasks:
    async def submit(self, body: TaskCreate, idempotency_key: str) -> tuple[Task, bool]:
        return TASK, True

    async def get(self, task_id: str) -> Task:
        return TASK

    async def list(self, cursor: str | None, limit: int, compute: str | None, state: TaskState | None, correlation_id: str | None) -> Page[Task]:
        return Page(items=(TASK,))

    async def cancel(self, task_id: str, idempotency_key: str) -> Task:
        return TASK

    async def result(self, task_id: str, wait_seconds: int) -> bytes | None:
        return b"\x00mock-result"

    async def expire(self) -> tuple[str, ...]:
        return ()


class MockExecutions:
    async def list(self, task_id: str) -> Page[Execution]:
        return Page(items=(EXECUTION,))

    async def get(self, task_id: str, ordinal: int) -> Execution:
        if ordinal != EXECUTION.ordinal:
            raise NotFoundError("no such execution", task=task_id, ordinal=ordinal)
        return EXECUTION

    async def create(self, task_id: str, body: ExecutionCreate, idempotency_key: str) -> Task:
        return TASK


class MockEvents:
    async def stream(
        self,
        last_event_id: str | None,
        compute: str | None,
        task: str | None,
        types: tuple[str, ...] | None,
    ) -> AsyncIterator[tuple[int, str, bytes]]:
        yield 1, "compute.provisioning", b'{"compute":"cmp_7f3a1c"}'
        yield 2, "node.ready", b'{"node":"nod_c19e40","rank":0}'
        yield 3, "task.succeeded", b'{"task":"tsk_9d21f0"}'


class MockProviders:
    async def create(self, body: ProviderCreate) -> Provider:
        return PROVIDER

    async def update(self, ref: str, body: ProviderCreate) -> Provider:
        return PROVIDER

    async def get(self, ref: str) -> Provider:
        return PROVIDER

    async def list(self) -> Page[Provider]:
        return Page(items=(PROVIDER,))

    async def delete(self, ref: str) -> None:
        return None


class MockOffers:
    async def list(
        self,
        provider: str | None,
        kind: str | None,
        accelerator: str | None,
        min_count: int | None,
        min_vram: float | None,
        max_price: float | None,
        refresh: bool,
    ) -> Page[Offer]:
        return Page(items=(OFFER,))


class MockReconciler:
    def __init__(self) -> None:
        self.reconciled: list[str] = []

    async def compute(self, compute_id: str) -> None:
        self.reconciled.append(f"compute:{compute_id}")

    async def observed(self, compute_id: str, node_id: str, state: str, error: str) -> None:
        self.reconciled.append(f"node:{node_id}:{state}")

    async def unsettled(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return (), ()


class MockDispatcher:
    def __init__(self) -> None:
        self.dispatched: list[str] = []

    async def task(self, task_id: str) -> None:
        self.dispatched.append(f"task:{task_id}")

    async def resume(self, compute_id: str) -> None:
        self.dispatched.append(f"resume:{compute_id}")

    async def stream(self, task_id: str) -> AsyncIterator[bytes]:
        self.dispatched.append(f"stream:{task_id}")
        return
        yield b""


class MockHealth:
    async def live(self) -> bool:
        return True

    async def ready(self) -> bool:
        return True

    async def dependencies(self) -> dict[str, DependencyState]:
        return {"store": "ok", "aws": "ok", "workers": "ok"}
