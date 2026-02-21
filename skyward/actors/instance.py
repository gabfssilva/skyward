from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any

from casty import ActorContext, ActorRef, Behavior, Behaviors

from skyward.actors.messages import (
    BootstrapDone,
    Bootstrapped,
    Bootstrapping,
    Execute,
    HeadAddressKnown,
    InstanceBecameReady,
    InstanceDied,
    InstanceMsg,
    Log,
    Metric,
    NodeInstance,
    Preempted,
    TaskResult,
    _bind_to_node,
    _BootstrapUploaded,
    _BootstrapUploadFailed,
    _Connected,
    _ConnectionFailed,
    _CorrelatedTaskResult,
    _LocalInstallDone,
    _PollResult,
    _PostBootstrapFailed,
    _SnapshotFailed,
    _SnapshotSaved,
    _UserCodeSyncDone,
    _WorkerFailed,
    _WorkerStarted,
)
from skyward.actors.streaming import instance_monitor
from skyward.infra.worker import (
    ExecuteTask as WorkerExecuteTask,
)
from skyward.infra.worker import (
    TaskFailed as WorkerTaskFailed,
)
from skyward.infra.worker import (
    TaskSucceeded as WorkerTaskSucceeded,
)
from skyward.infra.worker import (
    _CompressedResult,
    _decompress_result,
)
from skyward.observability.logger import logger
from skyward.providers.provider import WarmableProvider


def instance_actor(
    instance_id: str,
    provider: Any,
    cluster: Any,
    spec: Any,
    node_id: int,
    parent: ActorRef,
    _skip_monitor: bool = False,
    ssh_timeout: float = 300.0,
    ssh_retry_interval: float = 5.0,
    poll_interval: float = 5.0,
    poll_timeout: float = 300.0,
) -> Behavior[InstanceMsg]:
    """polling → connecting → bootstrapping → post_bootstrap → ready → joined."""

    log = logger.bind(actor="instance", instance_id=instance_id)
    provider_name = spec.provider or "aws"

    def polling() -> Behavior[InstanceMsg]:
        async def setup(ctx: ActorContext[InstanceMsg]) -> Behavior[InstanceMsg]:
            log.info("Polling for instance readiness")
            start_time = asyncio.get_event_loop().time()

            async def _do_poll() -> _PollResult:
                _, inst = await provider.get_instance(cluster, instance_id)
                return _PollResult(instance=inst)

            ctx.pipe_to_self(
                _do_poll(),
                on_failure=lambda e: _PollResult(instance=None),
            )

            return _polling_receive(ctx, start_time, head_info=None)

        return Behaviors.setup(setup)

    def _polling_receive(
        ctx: ActorContext[InstanceMsg],
        start_time: float,
        head_info: HeadAddressKnown | None,
    ) -> Behavior[InstanceMsg]:
        async def receive(
            ctx: ActorContext[InstanceMsg], msg: InstanceMsg,
        ) -> Behavior[InstanceMsg]:
            match msg:
                case HeadAddressKnown() as h:
                    return _polling_receive(ctx, start_time, h)
                case _PollResult(instance=inst) if (
                    inst and inst.status == "provisioned" and inst.ip
                ):
                    ni = _bind_to_node(inst, node_id, provider_name, cluster)
                    log.info("Instance ready at {ip}", ip=inst.ip)
                    return connecting(inst.ip, ni, head_info)
                case _PollResult():
                    elapsed = asyncio.get_event_loop().time() - start_time
                    if elapsed > poll_timeout:
                        parent.tell(InstanceDied(
                            instance_id=instance_id,
                            reason=f"Instance not ready within {poll_timeout}s",
                        ))
                        return Behaviors.stopped()

                    async def _poll_after_delay() -> _PollResult:
                        await asyncio.sleep(poll_interval)
                        _, inst = await provider.get_instance(cluster, instance_id)
                        return _PollResult(instance=inst)

                    ctx.pipe_to_self(
                        _poll_after_delay(),
                        on_failure=lambda e: _PollResult(instance=None),
                    )
                    return _polling_receive(ctx, start_time, head_info)
                case Preempted():
                    log.warning("Preempted during polling")
                    parent.tell(InstanceDied(instance_id=instance_id, reason="preempted"))
                    return Behaviors.stopped()
            return Behaviors.same()
        return Behaviors.receive(receive)

    def connecting(
        ip: str,
        ni: NodeInstance,
        head_info: HeadAddressKnown | None,
    ) -> Behavior[InstanceMsg]:
        async def setup(ctx: ActorContext[InstanceMsg]) -> Behavior[InstanceMsg]:
            log.info("Opening SSH tunnel to {ip}", ip=ip)
            ctx.pipe_to_self(
                _open_tunnel(ni, cluster),
                mapper=lambda result: _Connected(transport=result[0], listener=result[1]),
                on_failure=lambda e: _ConnectionFailed(error=str(e)),
            )
            return _connecting_receive(ctx, ip, ni, head_info)

        return Behaviors.setup(setup)

    def _connecting_receive(
        ctx: ActorContext[InstanceMsg],
        ip: str,
        ni: NodeInstance,
        head_info: HeadAddressKnown | None,
    ) -> Behavior[InstanceMsg]:
        async def receive(
            ctx: ActorContext[InstanceMsg], msg: InstanceMsg,
        ) -> Behavior[InstanceMsg]:
            match msg:
                case _Connected(transport=transport, listener=listener):
                    log.info("SSH tunnel established")
                    return bootstrapping(ip, ctx, ni, transport, listener, head_info)
                case _ConnectionFailed(error=error):
                    log.error("SSH connection failed: {error}", error=error)
                    parent.tell(InstanceDied(instance_id=instance_id, reason=error))
                    return Behaviors.stopped()
                case HeadAddressKnown() as h:
                    return _connecting_receive(ctx, ip, ni, h)
                case Preempted():
                    parent.tell(InstanceDied(instance_id=instance_id, reason="preempted"))
                    return Behaviors.stopped()
            return Behaviors.same()
        return Behaviors.receive(receive)

    def bootstrapping(
        ip: str, ctx: ActorContext[InstanceMsg],
        ni: NodeInstance,
        transport: Any, listener: Any,
        head_info: HeadAddressKnown | None,
    ) -> Behavior[InstanceMsg]:
        log.info("Starting bootstrap")
        if not _skip_monitor and ni.ssh_user and ni.ssh_key_path:
            ctx.spawn(
                instance_monitor(
                    info=ni,
                    ssh_user=ni.ssh_user,
                    ssh_key_path=ni.ssh_key_path,
                    event_listener=parent,
                    reply_to=ctx.self,
                    ssh_timeout=ssh_timeout,
                    ssh_retry_interval=ssh_retry_interval,
                ),
                f"monitor-{instance_id}",
            )

        if cluster.prebaked:
            log.info("Prebaked image detected, skipping bootstrap")
            return _start_post_bootstrap(ip, ctx, transport, listener, ni, head_info)

        ctx.pipe_to_self(
            _run_bootstrap(transport, ni, cluster, spec),
            mapper=lambda _: _BootstrapUploaded(),
            on_failure=lambda e: _BootstrapUploadFailed(error=str(e)),
        )

        async def receive(
            ctx: ActorContext[InstanceMsg], msg: InstanceMsg,
        ) -> Behavior[InstanceMsg]:
            match msg:
                case HeadAddressKnown() as h:
                    return bootstrapping(ip, ctx, ni, transport, listener, h)
                case BootstrapDone(success=True, instance=done_info):
                    log.info("Bootstrap completed successfully")
                    final_ni = done_info or ni
                    match provider:
                        case WarmableProvider() if node_id == 0:
                            log.info("Snapshotting provider image...")
                            ctx.pipe_to_self(
                                provider.save(cluster),
                                mapper=lambda _: _SnapshotSaved(),
                                on_failure=lambda e: _SnapshotFailed(error=str(e)),
                            )
                    return _start_post_bootstrap(
                        ip, ctx, transport, listener, final_ni, head_info,
                    )
                case BootstrapDone(success=False, error=error):
                    log.error("Bootstrap failed: {error}", error=error)
                    reason = error or "bootstrap failed"
                    await _cleanup_transport(transport, listener)
                    parent.tell(InstanceDied(instance_id=instance_id, reason=reason))
                    return Behaviors.stopped()
                case _BootstrapUploaded():
                    log.info("Bootstrap script uploaded and started")
                    return Behaviors.same()
                case _BootstrapUploadFailed(error=error):
                    log.error("Bootstrap upload failed: {error}", error=error)
                    await _cleanup_transport(transport, listener)
                    parent.tell(InstanceDied(instance_id=instance_id, reason=error))
                    return Behaviors.stopped()
                case Bootstrapping():
                    return Behaviors.same()
                case Bootstrapped():
                    return _start_post_bootstrap(ip, ctx, transport, listener, ni, head_info)
                case Preempted():
                    log.warning("Preempted during bootstrap")
                    await _cleanup_transport(transport, listener)
                    parent.tell(InstanceDied(instance_id=instance_id, reason="preempted"))
                    return Behaviors.stopped()
            return Behaviors.same()
        return Behaviors.receive(receive)

    def _start_post_bootstrap(
        ip: str, ctx: ActorContext[InstanceMsg],
        transport: Any, listener: Any,
        ni: NodeInstance,
        head_info: HeadAddressKnown | None,
    ) -> Behavior[InstanceMsg]:
        if spec.image and getattr(spec.image, "skyward_source", None) == "local":
            ctx.pipe_to_self(
                _install_local_skyward(ni, cluster),
                mapper=lambda _, m=ni: _LocalInstallDone(instance=m),
                on_failure=lambda e: _PostBootstrapFailed(error=str(e)),
            )
            return post_bootstrap(ip, transport, listener, ni, head_info)

        if spec.image and getattr(spec.image, "includes", None):
            ctx.pipe_to_self(
                _sync_user_code(ni, spec, cluster),
                mapper=lambda _, m=ni: _UserCodeSyncDone(instance=m),
                on_failure=lambda e: _PostBootstrapFailed(error=str(e)),
            )
            return post_bootstrap(ip, transport, listener, ni, head_info)

        return ready(ip, transport, listener, ni, head_info)

    def post_bootstrap(
        ip: str,
        transport: Any, listener: Any,
        ni: NodeInstance,
        head_info: HeadAddressKnown | None,
    ) -> Behavior[InstanceMsg]:
        async def receive(
            ctx: ActorContext[InstanceMsg], msg: InstanceMsg,
        ) -> Behavior[InstanceMsg]:
            match msg:
                case HeadAddressKnown() as h:
                    return post_bootstrap(ip, transport, listener, ni, h)
                case _LocalInstallDone(instance=info):
                    if spec.image and getattr(spec.image, "includes", None):
                        ctx.pipe_to_self(
                            _sync_user_code(info, spec, cluster),
                            mapper=lambda _, m=info: _UserCodeSyncDone(instance=m),
                            on_failure=lambda e: _PostBootstrapFailed(error=str(e)),
                        )
                        return Behaviors.same()
                    return ready(ip, transport, listener, ni, head_info)
                case _UserCodeSyncDone():
                    return ready(ip, transport, listener, ni, head_info)
                case _PostBootstrapFailed(error=err):
                    await _cleanup_transport(transport, listener)
                    parent.tell(InstanceDied(instance_id=instance_id, reason=err))
                    return Behaviors.stopped()
                case _SnapshotSaved():
                    log.info("Snapshot saved")
                case _SnapshotFailed(error=error):
                    log.warning("Snapshot failed: {error}", error=error)
                case Preempted():
                    await _cleanup_transport(transport, listener)
                    parent.tell(InstanceDied(instance_id=instance_id, reason="preempted"))
                    return Behaviors.stopped()
            return Behaviors.same()
        return Behaviors.receive(receive)

    def ready(
        ip: str,
        transport: Any, listener: Any,
        ni: NodeInstance,
        head_info: HeadAddressKnown | None,
    ) -> Behavior[InstanceMsg]:
        is_head = node_id == 0

        if is_head:
            head_private = ni.instance.private_ip or ni.instance.ip or ""
            parent.tell(HeadAddressKnown(
                head_addr=head_private,
                casty_port=25520,
                num_nodes=spec.nodes,
                concurrency=spec.concurrency,
            ))
            log.info("Starting worker (role=head)")
            return _start_worker(ip, transport, listener, ni, head_info)

        if head_info is not None:
            log.info("Starting worker (role=worker)")
            return _start_worker(ip, transport, listener, ni, head_info)

        log.info("Waiting for head address before starting worker")
        return waiting_for_head(ip, transport, listener, ni)

    def waiting_for_head(
        ip: str,
        transport: Any, listener: Any,
        ni: NodeInstance,
    ) -> Behavior[InstanceMsg]:
        async def receive(
            ctx: ActorContext[InstanceMsg], msg: InstanceMsg,
        ) -> Behavior[InstanceMsg]:
            match msg:
                case HeadAddressKnown() as h:
                    log.info("Head address received, starting worker")
                    return _start_worker(ip, transport, listener, ni, h)
                case _SnapshotSaved():
                    log.info("Snapshot saved")
                case _SnapshotFailed(error=error):
                    log.warning("Snapshot failed: {error}", error=error)
                case Preempted():
                    log.warning("Preempted while waiting for head")
                    await _cleanup_transport(transport, listener)
                    parent.tell(InstanceDied(instance_id=instance_id, reason="preempted"))
                    return Behaviors.stopped()
            return Behaviors.same()
        return Behaviors.receive(receive)

    def _start_worker(
        ip: str,
        transport: Any, listener: Any,
        ni: NodeInstance,
        head_info: HeadAddressKnown | None,
    ) -> Behavior[InstanceMsg]:
        async def setup(ctx: ActorContext[InstanceMsg]) -> Behavior[InstanceMsg]:
            ctx.pipe_to_self(
                _do_start_worker(transport, listener, ni, head_info, node_id, cluster, spec),
                mapper=lambda result: _WorkerStarted(client=result[0], worker_ref=result[1]),
                on_failure=lambda e: _WorkerFailed(error=str(e)),
            )
            return _starting_worker_receive(ctx, ip, transport, listener, ni)

        return Behaviors.setup(setup)

    def _starting_worker_receive(
        ctx: ActorContext[InstanceMsg],
        ip: str,
        transport: Any, listener: Any,
        ni: NodeInstance,
    ) -> Behavior[InstanceMsg]:
        async def receive(
            ctx: ActorContext[InstanceMsg], msg: InstanceMsg,
        ) -> Behavior[InstanceMsg]:
            match msg:
                case _WorkerStarted(client=client, worker_ref=wref):
                    log.info("Worker joined cluster successfully")
                    parent.tell(InstanceBecameReady(
                        instance_id=instance_id, ip=ip,
                        node_instance=ni,
                    ))
                    return joined(ip, client, wref, transport, listener)
                case _WorkerFailed(error=error):
                    log.error("Worker failed to start: {error}", error=error)
                    await _cleanup_transport(transport, listener)
                    parent.tell(InstanceDied(instance_id=instance_id, reason=error))
                    return Behaviors.stopped()
                case _SnapshotSaved():
                    log.info("Snapshot saved")
                case _SnapshotFailed(error=error):
                    log.warning("Snapshot failed: {error}", error=error)
                case Preempted():
                    log.warning("Preempted while starting worker")
                    await _cleanup_transport(transport, listener)
                    parent.tell(InstanceDied(instance_id=instance_id, reason="preempted"))
                    return Behaviors.stopped()
            return Behaviors.same()
        return Behaviors.receive(receive)

    def joined(
        ip: str,
        client: Any, worker_ref: Any,
        transport: Any, listener: Any,
    ) -> Behavior[InstanceMsg]:
        async def receive(
            ctx: ActorContext[InstanceMsg], msg: InstanceMsg,
        ) -> Behavior[InstanceMsg]:
            match msg:
                case Execute(fn=fn, args=args, kwargs=kwargs, task_id=tid, timeout=timeout):
                    log.debug("Executing task {tid}", tid=tid)
                    ctx.pipe_to_self(
                        _execute_with_streaming(client, worker_ref, fn, args, kwargs, timeout),
                        mapper=lambda result, _tid=tid: _CorrelatedTaskResult(  # type: ignore[return-value]
                            task_id=_tid,
                            value=(
                                result.result
                                if isinstance(result, WorkerTaskSucceeded)
                                else RuntimeError(result.error)
                            ),
                            node_id=result.node_id,
                            error=not isinstance(result, WorkerTaskSucceeded),
                        ),
                        on_failure=lambda e, _tid=tid: _CorrelatedTaskResult(  # type: ignore[return-value]
                            task_id=_tid, value=RuntimeError(str(e)), node_id=0, error=True,
                        ),
                    )
                    return Behaviors.same()
                case _CorrelatedTaskResult(task_id=tid, value=value, node_id=nid, error=is_err):
                    if is_err:
                        log.warning("Task {tid} failed: {error}", tid=tid, error=value)
                    else:
                        log.debug("Task {tid} succeeded", tid=tid)
                    parent.tell(TaskResult(value=value, node_id=nid, task_id=tid))
                    return Behaviors.same()
                case Log() | Metric():
                    pass
                case _SnapshotSaved():
                    log.info("Snapshot saved")
                case _SnapshotFailed(error=error):
                    log.warning("Snapshot failed: {error}", error=error)
                case Preempted():
                    log.warning("Preempted while joined")
                    await _cleanup(client, transport, listener)
                    parent.tell(InstanceDied(instance_id=instance_id, reason="preempted"))
                    return Behaviors.stopped()
            return Behaviors.same()
        return Behaviors.receive(receive)

    return polling()


async def _open_tunnel(ni: NodeInstance, cluster: Any) -> tuple[Any, Any]:
    from skyward.infra.ssh import SSHTransport

    transport = SSHTransport(
        host=ni.instance.ip or "",
        user=ni.ssh_user or cluster.ssh_user,
        key_path=ni.ssh_key_path or cluster.ssh_key_path,
        port=ni.instance.ssh_port,
    )
    await transport.connect()
    listener = await transport._conn.forward_local_port(  # type: ignore[union-attr]
        "", 0, "127.0.0.1", 25520,
    )
    return transport, listener


async def _do_start_worker(
    transport: Any,
    listener: Any,
    ni: NodeInstance,
    head_info: HeadAddressKnown | None,
    node_id: int,
    cluster: Any,
    spec: Any,
) -> tuple[Any, Any]:
    import logging as _logging

    from casty import ClusterClient

    from skyward.infra.executor import _wait_for_workers
    from skyward.infra.serialization import check_python_version, serialize
    from skyward.infra.worker import ExecuteTask as WorkerExecuteTask
    from skyward.providers.bootstrap.compose import EMIT_SH_PATH, SKYWARD_DIR
    from skyward.providers.common import detect_network_interface

    _logging.getLogger("casty").setLevel(_logging.ERROR)

    log = logger.bind(actor="instance", instance_id=ni.instance.id)

    private_ip = ni.instance.private_ip or ni.instance.ip or ""
    casty_port = head_info.casty_port if head_info else 25520
    num_nodes = head_info.num_nodes if head_info else spec.nodes
    concurrency = head_info.concurrency if head_info else spec.concurrency

    seeds = f"{head_info.head_addr}:{casty_port}" if head_info and node_id != 0 else ""

    venv_dir = f"{SKYWARD_DIR}/.venv"
    python_bin = f"{venv_dir}/bin/python"
    ssh_user = ni.ssh_user or cluster.ssh_user
    use_sudo = ssh_user != "root"

    host = private_ip if node_id != 0 else (head_info.head_addr if head_info else private_ip)

    seeds_arg = f"--seeds {seeds} " if seeds else ""
    casty_cmd = (
        f'nohup {python_bin} -c "from skyward.infra.worker import cli; cli()" '
        f"--node-id {node_id} --port {casty_port} "
        f"--num-nodes {num_nodes} --host {host} "
        f"--workers-per-node {concurrency} "
        f"{seeds_arg}"
        f"> /var/log/casty.log 2>&1 & echo $!"
    )
    tail_inner = (
        f"source {EMIT_SH_PATH} && "
        f"tail -f /var/log/casty.log 2>/dev/null | while IFS= read -r line; do "
        f'emit_console "$line"; done'
    )
    tail_cmd = f"nohup bash -c '{tail_inner}' </dev/null >/dev/null 2>&1 &"

    if use_sudo:
        casty_cmd = f"sudo bash -c '{casty_cmd}'"
        tail_cmd = f"sudo {tail_cmd}"

    exit_code, stdout, stderr = await transport.run(casty_cmd, timeout=60.0)
    if exit_code != 0:
        raise RuntimeError(f"Failed to start Casty node {node_id}: {stderr}")
    await transport.run(tail_cmd, timeout=10.0)
    log.debug("Casty worker started, PID: {pid}", pid=stdout.strip())

    local_port = listener.get_port()
    address_map = {(private_ip, casty_port): ("127.0.0.1", local_port)}
    host_to_node = {private_ip: node_id}

    client = ClusterClient(
        contact_points=[(private_ip, casty_port)],
        system_name="skyward",
        address_map=address_map,
    )
    await client.__aenter__()

    worker_refs = await _wait_for_workers(client, 1, host_to_node, timeout=120.0)
    worker_ref = worker_refs[node_id]

    network_iface = await detect_network_interface(transport)

    if spec.image:
        check_python_version(spec.image.python)

    from skyward.providers.pool_info import build_pool_info

    all_nodes = spec.nodes
    accel = ni.instance.offer.instance_type.accelerator
    accelerator_count = accel.count if accel else 1
    total_accelerators = accelerator_count * all_nodes
    head_addr = head_info.head_addr if head_info else private_ip

    pool_info = build_pool_info(
        node=node_id,
        total_nodes=all_nodes,
        accelerator_count=accelerator_count,
        total_accelerators=total_accelerators,
        head_addr=head_addr,
        head_port=29500,
        job_id=cluster.id,
        peers=[],
        accelerator_type=getattr(spec, "accelerator_name", None),
        placement_group=network_iface or ni.network_interface or None,
        worker=0,
        workers_per_node=1,
    )

    image_env = dict(spec.image.env) if spec.image and spec.image.env else {}

    def setup_env(info_json: str, extra: dict[str, str]) -> str:
        import os
        os.environ["COMPUTE_POOL"] = info_json
        for k, v in extra.items():
            os.environ[k] = v
        return "ok"

    pool_json = pool_info.model_dump_json()
    payload = {"fn": setup_env, "args": (pool_json, image_env), "kwargs": {}}
    fn_bytes = await asyncio.to_thread(serialize, payload)
    await client.ask(
        worker_ref,
        lambda reply_to: WorkerExecuteTask(fn_bytes=fn_bytes, reply_to=reply_to),
        timeout=60.0,
    )

    return client, worker_ref


async def _cleanup(client: Any, transport: Any, listener: Any) -> None:
    with suppress(Exception):
        await client.__aexit__(None, None, None)
    await _cleanup_transport(transport, listener)


async def _cleanup_transport(transport: Any, listener: Any) -> None:
    with suppress(Exception):
        listener.close()
    with suppress(Exception):
        await transport.close()


async def _install_local_skyward(ni: NodeInstance, cluster: Any) -> None:
    from skyward.providers._bootstrap_ssh import install_local_skyward, wait_for_ssh

    transport = await wait_for_ssh(
        host=ni.instance.ip or "", user=cluster.ssh_user, key_path=cluster.ssh_key_path,
        port=ni.instance.ssh_port,
    )
    try:
        await install_local_skyward(
            transport=transport, ni=ni, use_sudo=cluster.use_sudo,
        )
    finally:
        await transport.close()


async def _sync_user_code(ni: NodeInstance, spec: Any, cluster: Any) -> None:
    from skyward.providers._bootstrap_ssh import sync_user_code

    await sync_user_code(
        host=ni.instance.ip or "", user=cluster.ssh_user, key_path=cluster.ssh_key_path,
        port=ni.instance.ssh_port, image=spec.image, use_sudo=cluster.use_sudo,
    )


async def _run_bootstrap(
    transport: Any, ni: NodeInstance, cluster: Any, spec: Any,
) -> None:
    from skyward.providers._bootstrap_ssh import run_bootstrap_via_ssh

    image = spec.image
    if not image:
        return

    ssh_user = ni.ssh_user or cluster.ssh_user
    use_sudo = ssh_user != "root"

    bootstrap_script = image.generate_bootstrap(ttl=0)

    await run_bootstrap_via_ssh(
        transport=transport,
        ni=ni,
        bootstrap_script=bootstrap_script,
        use_sudo=use_sudo,
    )


async def _execute_with_streaming(
    client: Any,
    worker_ref: Any,
    fn: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    timeout: float,
) -> WorkerTaskSucceeded | WorkerTaskFailed:
    from skyward.infra.serialization import serialize
    from skyward.infra.streaming import _stream_param_indices, _StreamHandle, _SyncSource

    indices = _stream_param_indices(fn)
    pump_tasks: list[asyncio.Task[None]] = []
    resolved_args = args
    stream_refs: tuple[tuple[int, Any], ...] = ()

    if indices:
        resolved_args, pump_tasks, stream_refs = await _setup_input_streams(
            client, args, indices,
        )

    payload = {"fn": fn, "args": resolved_args, "kwargs": kwargs}
    fn_bytes = await asyncio.to_thread(serialize, payload)
    log = logger.bind(component="execute_streaming")
    log.debug("Sending to worker, {n} bytes, streams={s}", n=len(fn_bytes), s=len(stream_refs))
    result = await client.ask(
        worker_ref,
        lambda rto: WorkerExecuteTask(
            fn_bytes=fn_bytes, reply_to=rto, input_streams=stream_refs,
        ),
        timeout=timeout,
    )
    log.debug("Worker replied: {t}", t=type(result).__name__)

    for t in pump_tasks:
        await t

    match result:
        case WorkerTaskSucceeded(result=_StreamHandle(producer_ref=pref)):
            source = await _resolve_output_stream(client, pref)
            loop = asyncio.get_running_loop()
            return WorkerTaskSucceeded(result=_SyncSource(source, loop), node_id=result.node_id)
        case WorkerTaskSucceeded(result=_CompressedResult() as compressed):
            decompressed = await asyncio.to_thread(_decompress_result, compressed)
            return WorkerTaskSucceeded(result=decompressed, node_id=result.node_id)
        case _:
            return result


async def _setup_input_streams(
    client: Any,
    args: tuple[Any, ...],
    indices: tuple[int, ...],
) -> tuple[tuple[Any, ...], list[asyncio.Task[None]], tuple[tuple[int, Any], ...]]:
    from uuid import uuid4

    from casty import GetSink, stream_producer

    resolved = list(args)
    pump_tasks: list[asyncio.Task[None]] = []
    stream_refs: list[tuple[int, Any]] = []
    loop = asyncio.get_running_loop()

    for i in indices:
        if i >= len(resolved):
            continue
        iterator = resolved[i]

        producer_ref = client.spawn(
            stream_producer(buffer_size=256),
            f"input-{uuid4().hex[:8]}",
        )
        sink = await client.ask(producer_ref, lambda r: GetSink(reply_to=r), timeout=10.0)
        resolved[i] = None
        stream_refs.append((i, producer_ref))

        async def _pump(_sink: Any = sink, _it: Any = iterator) -> None:
            def _drain() -> None:
                try:
                    for elem in _it:
                        fut = asyncio.run_coroutine_threadsafe(_sink.put(elem), loop)
                        fut.result(timeout=300.0)
                finally:
                    fut = asyncio.run_coroutine_threadsafe(_sink.complete(), loop)
                    fut.result(timeout=60.0)
            await asyncio.to_thread(_drain)

        pump_tasks.append(asyncio.create_task(_pump()))

    return tuple(resolved), pump_tasks, tuple(stream_refs)


async def _resolve_output_stream(client: Any, producer_ref: Any) -> Any:
    from uuid import uuid4

    from casty import GetSource, stream_consumer

    consumer_ref = client.spawn(
        stream_consumer(producer_ref, timeout=60.0, initial_demand=16),
        f"out-consumer-{uuid4().hex[:8]}",
    )
    return await client.ask(consumer_ref, lambda r: GetSource(reply_to=r), timeout=10.0)
