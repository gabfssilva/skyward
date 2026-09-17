"""The ``v1`` documents, made out of what the daemon read, and the records made out of what a caller sent.

A record carries what the daemon needs — a machine's binding with the password a
provider handed over, the counters only the reconciler reads — and a resource carries
what a client draws. Nothing reaches a client except through here, which is what
keeps the first from leaking into the second.
"""

from datetime import datetime

import msgspec
from msgspec import UNSET

from skyward.api import v1
from skyward.server.application.reading import Code, ComputeReading, Finished, NodeReading, TaskReading
from skyward.shared.schemas import Error, TaskCounts


def recast[T](value: object, kind: type[T]) -> T:
    """The same document under another type, for the values whose wire shape the two agree on.

    Validated on the way, so a field one side has and the other lacks is an error
    here and not a silently dropped value.
    """
    return msgspec.convert(msgspec.to_builtins(value, builtin_types=(bytes, datetime)), kind)


def compute(reading: ComputeReading) -> v1.ComputeResource:
    compute, offer = reading.compute, reading.compute.offer
    return v1.ComputeResource(
        id=compute.id,
        name=compute.name,
        revision=compute.revision,
        generation=compute.generation,
        created_at=compute.created_at,
        status=v1.ComputeStatus(
            state=compute.status.state,
            observed_generation=compute.status.observed_generation,
            last_error=_error(compute.status.last_error),
        ),
        spec=recast(compute.spec, v1.ComputeSpec),
        provider=v1.ProviderSummary(id=offer.provider_id, name=offer.provider_name, kind=offer.kind) if offer else None,
        offer=recast(offer, v1.OfferResource) if offer else None,
        lease=v1.LeaseResource(owner=compute.lease.owner, expires_at=compute.lease.expires_at),
        cost=compute.cost,
        rate=reading.rate,
        tasks=_counts(compute.tasks, reading),
        placement=recast(compute.placement, v1.Refusal) if compute.placement else None,
        ended=recast(compute.ended, v1.Ending) if compute.ended else None,
        nodes=tuple(node(each) for each in reading.nodes),
        utilization=UNSET if reading.utilization is None else recast(reading.utilization, v1.Utilization),
    )


def node(reading: NodeReading) -> v1.NodeResource:
    node, machine = reading.node, reading.machine
    return v1.NodeResource(
        id=node.id,
        rank=node.rank,
        generation=node.generation,
        created_at=node.created_at,
        state=node.state,
        desired=node.desired,
        machine=node.machine,
        address=node.address,
        ssh=v1.Ssh(host=machine.host, port=machine.port, user=machine.user) if machine and machine.host else None,
        accelerator=node.accelerator,
        market=node.market,
        price_per_hour=node.price_per_hour,
        billing_unit=node.billing_unit,
        launched_at=node.launched_at,
        terminated_at=node.terminated_at,
        last_error=_error(node.last_error),
        progress=(
            v1.Progress(step=machine.progress, completion=machine.completion)
            if machine and machine.progress is not None and node.state == "provisioning"
            else None
        ),
        busy=reading.busy,
        metrics=UNSET if reading.metrics is None else {sample.name: v1.Gauge(at=sample.at, value=sample.value) for sample in reading.metrics},
        phases=(
            UNSET
            if reading.phases is None
            else tuple(v1.Phase(name=mark.phase, state=mark.event, at=mark.at, error=mark.error) for mark in reading.phases)
        ),
        running=(
            UNSET
            if reading.running is None
            else tuple(v1.Running(task=held.task, ordinal=held.ordinal, function=_function(held.code), started_at=held.started_at) for held in reading.running)
        ),
        tail=UNSET if reading.tail is None else reading.tail,
    )


def task(reading: TaskReading) -> v1.TaskResource:
    task = reading.task
    return v1.TaskResource(
        id=task.id,
        compute=v1.ComputeSummary(id=task.compute_id, name=reading.compute),
        generation=task.generation,
        function=_function(reading.code),
        args_sha256=task.args_sha256,
        dispatch=task.dispatch,
        state=task.state,
        retry=task.retry,
        executions=tuple(recast(execution, v1.ExecutionResource) for execution in task.executions),
        submitted_at=task.submitted_at,
        finished_at=task.finished_at,
        rank=task.rank,
        correlation_id=task.correlation_id,
        queue_timeout_seconds=task.queue_timeout_seconds,
        run_timeout_seconds=task.run_timeout_seconds,
        result_sha256=task.result_sha256,
    )


def _counts(counts: TaskCounts, reading: ComputeReading) -> v1.TaskCounts:
    return v1.TaskCounts(
        queued=counts.queued,
        running=counts.running,
        succeeded=counts.succeeded,
        failed=counts.failed,
        cancelled=counts.cancelled,
        timed_out=counts.timed_out,
        indeterminate=counts.indeterminate,
        latest=(
            UNSET
            if reading.latest is None
            else v1.LatestTasks(succeeded=_summary(reading.latest.succeeded), failed=_summary(reading.latest.failed))
        ),
        pace=(
            UNSET
            if reading.pace is None
            else v1.Pace(finished_last_hour=reading.pace.finished, mean_seconds=reading.pace.mean_seconds)
        ),
    )


def _summary(finished: Finished | None) -> v1.TaskSummary | None:
    if finished is None:
        return None
    task = finished.task
    return v1.TaskSummary(
        id=task.id,
        function=_function(finished.code),
        state=task.state,
        finished_at=task.finished_at,
        error=_error(next((execution.error for execution in reversed(task.executions) if execution.error), None)),
    )


def _function(code: Code) -> v1.FunctionSummary:
    function = code.function
    return v1.FunctionSummary(sha256=code.sha256, name=function.name if function else None, version=function.version if function else None)


def _error(error: Error | None) -> v1.Error | None:
    return recast(error, v1.Error) if error else None
