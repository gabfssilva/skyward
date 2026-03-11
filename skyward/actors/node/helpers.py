from __future__ import annotations

import asyncio
from typing import Any

from casty import ActorRef

from skyward.actors.messages import NodeInstance
from skyward.infra.ssh_actor import transport_run, transport_write_bytes, transport_write_file
from skyward.observability.logger import logger

from .state import check_python_version


def resolve_ssh_user(ni: NodeInstance, cluster: Any) -> tuple[str, str]:
    """Return (ssh_user, sudo_prefix) for remote commands."""
    ssh_user = ni.ssh_user or cluster.ssh_user
    sudo = "sudo " if ssh_user != "root" else ""
    return ssh_user, sudo


async def do_start_worker(
    transport_ref: ActorRef,
    local_port: int,
    ni: NodeInstance,
    head_info: Any,
    node_id: int,
    cluster: Any,
    spec: Any,
) -> tuple[int, str]:
    import logging as _logging

    from skyward.providers.bootstrap.compose import EMIT_SH_PATH, SKYWARD_DIR

    _logging.getLogger("casty").setLevel(_logging.ERROR)

    log = logger.bind(actor="node", instance_id=ni.instance.id)

    private_ip = ni.instance.private_ip or ni.instance.ip or ""
    casty_port = head_info.casty_port if head_info else 25520
    num_nodes = head_info.num_nodes if head_info else spec.nodes
    concurrency = head_info.worker_concurrency if head_info else spec.worker.concurrency
    executor = head_info.worker_executor if head_info else spec.worker.resolved_executor

    seeds = f"{head_info.head_addr}:{casty_port}" if head_info and node_id != 0 else ""

    venv_dir = f"{SKYWARD_DIR}/.venv"
    python_bin = f"{venv_dir}/bin/python"
    ssh_user, _ = resolve_ssh_user(ni, cluster)
    use_sudo = ssh_user != "root"

    host = private_ip if node_id != 0 else (head_info.head_addr if head_info else private_ip)

    from skyward.api.plugin import LaunchCommand, LaunchContext

    env_vars: dict[str, str] = {
        "SKYWARD_NODE_ID": str(node_id),
        "SKYWARD_PORT": str(casty_port),
        "SKYWARD_NUM_NODES": str(num_nodes),
        "SKYWARD_HOST": host,
        "SKYWARD_WORKERS_PER_NODE": str(concurrency),
        "SKYWARD_WORKER_EXECUTOR": executor,
    }
    if seeds:
        env_vars["SKYWARD_SEEDS"] = seeds

    launch_cmd = LaunchCommand(
        entrypoint=f'{python_bin} -c "from skyward.infra.worker import cli; cli()"',
    )
    launch_ctx = LaunchContext(
        node_id=node_id,
        num_nodes=num_nodes,
        head_addr=head_info.head_addr if head_info else private_ip,
        casty_port=casty_port,
        private_ip=private_ip,
        python_bin=python_bin,
        venv_dir=venv_dir,
        concurrency=concurrency,
        executor=executor,
    )

    for plugin in spec.plugins:
        if plugin.launcher is not None:
            launch_cmd = plugin.launcher(launch_cmd, launch_ctx)

    env_str = " ".join(f"{k}={v}" for k, v in env_vars.items())
    casty_cmd = f"nohup env {env_str} {launch_cmd.render()} > /var/log/casty.log 2>&1 & echo $!"
    tail_inner = (
        f"source {EMIT_SH_PATH} && "
        f"tail -f /var/log/casty.log 2>/dev/null "
        f'| tr "\\r" "\\n" '
        f'| sed "s/\\x1b\\[[0-9;?]*[a-zA-Z]//g; s/[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f]//g" '
        f"| while IFS= read -r line; do "
        f'[ -n "$line" ] && emit_console "$line"; done'
    )
    tail_cmd = f"nohup bash -c '{tail_inner}' </dev/null >/dev/null 2>&1 &"

    if use_sudo:
        casty_cmd = f"sudo bash -c '{casty_cmd}'"
        tail_cmd = f"sudo {tail_cmd}"

    log.info("Launch command: {cmd}", cmd=launch_cmd.render())
    exit_code, stdout, stderr = await transport_run(transport_ref, casty_cmd, timeout=60.0)
    if exit_code != 0:
        raise RuntimeError(f"Failed to start Casty node {node_id}: {stderr}")

    pid = stdout.strip()
    await transport_run(transport_ref, tail_cmd, timeout=10.0)
    log.info("Casty worker started, PID: {pid}", pid=pid)


    if spec.image:
        check_python_version(spec.image.python)

    return local_port, private_ip


async def install_local_skyward(transport_ref: ActorRef, ni: NodeInstance, cluster: Any) -> None:
    from skyward.providers.common import _build_wheel_install_script, build_wheel

    log = logger.bind(component="bootstrap_ssh")
    log.info("Building local skyward wheel")
    wheel_path = await asyncio.to_thread(build_wheel)

    install_script = _build_wheel_install_script(wheel_name=wheel_path.name)

    def _read_wheel() -> tuple[int, bytes]:
        return wheel_path.stat().st_size, wheel_path.read_bytes()

    wheel_size, wheel_data = await asyncio.to_thread(_read_wheel)
    log.debug("Wheel size: {size:.1f} KB", size=wheel_size / 1024)
    log.info("Uploading wheel {name}", name=wheel_path.name)
    await transport_write_bytes(transport_ref, f"/tmp/{wheel_path.name}", wheel_data)

    await transport_write_file(transport_ref, "/tmp/.install-wheel.sh", install_script)

    _, sudo = resolve_ssh_user(ni, cluster)
    log.info("Running wheel install script on {iid}", iid=ni.instance.id)
    exit_code, stdout, stderr = await transport_run(
        transport_ref, f"{sudo}bash /tmp/.install-wheel.sh", timeout=180.0,
    )
    log.debug("Install script output:\n{out}", out=stdout)

    if exit_code != 0:
        raise RuntimeError(f"Wheel install failed (exit {exit_code}): {stderr or stdout}")

    log.info("Local skyward wheel installed on {iid}", iid=ni.instance.id)


async def sync_user_code(transport_ref: ActorRef, ni: NodeInstance, spec: Any, cluster: Any) -> None:
    from skyward.providers.bootstrap.compose import SKYWARD_DIR
    from skyward.providers.common import build_user_code_tarball

    image = spec.image
    includes = getattr(image, "includes", ())
    if not includes:
        return

    excludes = getattr(image, "excludes", ())
    tarball = build_user_code_tarball(includes=includes, excludes=excludes)

    _, sudo = resolve_ssh_user(ni, cluster)
    remote_tar = "/tmp/_user_code.tar.gz"
    site_packages = f"{SKYWARD_DIR}/.venv/lib/python*/site-packages"

    log = logger.bind(component="bootstrap_ssh")
    log.info("Uploading user code ({size:.1f} KB)", size=len(tarball) / 1024)
    await transport_write_bytes(transport_ref, remote_tar, tarball)

    exit_code, stdout, stderr = await transport_run(
        transport_ref,
        f"{sudo}bash -c 'SP=$(echo {site_packages}); "
        f"tar xzf {remote_tar} -C $SP && rm -f {remote_tar}'",
        timeout=60.0,
    )

    if exit_code != 0:
        raise RuntimeError(f"User code extraction failed (exit {exit_code}): {stderr or stdout}")

    log.info("User code uploaded and extracted to site-packages")


async def run_bootstrap(
    transport_ref: ActorRef,
    ni: NodeInstance,
    cluster: Any,
    spec: Any,
) -> None:
    import base64

    image = spec.image
    if not image:
        return

    _, sudo = resolve_ssh_user(ni, cluster)

    postamble_ops: list = []
    if cluster.resolved_volumes:
        from skyward.providers.bootstrap import mount_volumes, phase

        postamble_ops.append(phase("volumes", mount_volumes(cluster.resolved_volumes)))
    for plugin in spec.plugins:
        if plugin.bootstrap is not None:
            postamble_ops.extend(plugin.bootstrap(cluster))

    postamble = postamble_ops if postamble_ops else None
    ttl = spec.ttl or 0
    shutdown_command = cluster.shutdown_command.format(instance_id=ni.instance.id)
    from skyward.core.spec import generate_bootstrap

    bootstrap_script = generate_bootstrap(
        image,
        ttl=ttl,
        shutdown_command=shutdown_command,
        postamble=postamble,
    )

    log = logger.bind(component="bootstrap_ssh")
    log.info(
        "Uploading bootstrap script to {iid} ({size:.1f} KB)",
        iid=ni.instance.id, size=len(bootstrap_script) / 1024,
    )

    encoded = base64.b64encode(bootstrap_script.encode()).decode()
    exit_code, _, stderr = await transport_run(
        transport_ref,
        f"{sudo}bash -c \""
        f"mkdir -p /opt/skyward && "
        f"echo '{encoded}' | base64 -d > /opt/skyward/bootstrap.sh && "
        f"chmod +x /opt/skyward/bootstrap.sh\"",
    )
    if exit_code != 0:
        raise RuntimeError(f"Bootstrap upload failed: {stderr}")

    log.info("Running bootstrap on {iid}", iid=ni.instance.id)
    await transport_run(
        transport_ref,
        f"{sudo}bash -c 'nohup /opt/skyward/bootstrap.sh > /opt/skyward/bootstrap.log 2>&1 &'",
    )

    log.info("Bootstrap started on {iid}", iid=ni.instance.id)


async def terminate_and_replace(provider: Any, cluster: Any, dead_id: str) -> Any:
    log = logger.bind(actor="node")
    try:
        await provider.terminate(cluster, (dead_id,))
    except Exception as e:
        log.warning("Failed to terminate dead instance {iid}: {err}", iid=dead_id, err=e)
    _, instances = await provider.provision(cluster, 1)
    if not instances:
        raise RuntimeError("Failed to provision replacement instance")
    return instances[0]


async def discover_own_worker(client: Any, ni: NodeInstance | None) -> Any:
    from skyward.infra.worker import WORKER_KEY

    private_ip = ni.instance.private_ip or ni.instance.ip or "" if ni else ""

    deadline = asyncio.get_event_loop().time() + 120.0
    while asyncio.get_event_loop().time() < deadline:
        listing = client.lookup(WORKER_KEY)
        for instance in listing.instances:
            if instance.node.host == private_ip:
                return instance.ref
        await asyncio.sleep(1.0)

    raise RuntimeError(f"Worker for {private_ip} not discovered within 120s")


async def setup_worker_env(
    client: Any,
    worker_ref: Any,
    pool_info_json: str,
    env_vars: dict[str, str],
    around_app_hooks: tuple[tuple[str, Any], ...] = (),
    around_process_hooks: tuple[tuple[str, Any], ...] = (),
) -> None:
    from skyward.infra.worker import EnterContext, SetProcessHooks

    _frozen_env = dict(env_vars)

    def make_env_cm(
        info_json: str = pool_info_json,
        extra: dict[str, str] = _frozen_env,
    ) -> Any:
        from contextlib import contextmanager

        @contextmanager
        def lifecycle() -> Any:
            import os

            os.environ["COMPUTE_POOL"] = info_json
            for k, v in extra.items():
                os.environ[k] = v
            yield

        return lifecycle()

    await client.ask(
        worker_ref,
        lambda rto: EnterContext(factory=make_env_cm, reply_to=rto),
        timeout=60.0,
    )

    if around_app_hooks:

        def make_hooks_cm(
            hooks: tuple[tuple[str, Any], ...] = around_app_hooks,
        ) -> Any:
            from contextlib import contextmanager

            @contextmanager
            def lifecycle() -> Any:
                from skyward.core.runtime import instance_info
                from skyward.plugins.state import cleanup, ensure_around_app

                info = instance_info()
                for name, factory in hooks:
                    ensure_around_app(name, factory, info)
                try:
                    yield
                finally:
                    cleanup()

            return lifecycle()

        await client.ask(
            worker_ref,
            lambda rto: EnterContext(factory=make_hooks_cm, reply_to=rto),
            timeout=60.0,
        )

    if around_process_hooks:
        await client.ask(
            worker_ref,
            lambda rto: SetProcessHooks(hooks=around_process_hooks, reply_to=rto),
            timeout=60.0,
        )


async def execute_with_streaming(
    client: Any,
    worker_ref: Any,
    fn: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    timeout: float,
) -> Any:
    from skyward.infra.streaming import _stream_param_indices, _StreamHandle, _SyncSource
    from skyward.infra.worker import ExecuteTask
    from skyward.infra.worker import TaskSucceeded as WorkerTaskSucceeded

    indices = _stream_param_indices(fn)
    pump_tasks: list[asyncio.Task[None]] = []
    resolved_args = args
    stream_refs: tuple[tuple[int, Any], ...] = ()

    if indices:
        resolved_args, pump_tasks, stream_refs = await _setup_input_streams(
            client,
            args,
            indices,
        )

    result = await asyncio.wait_for(
        client.ask(
            worker_ref,
            lambda rto: ExecuteTask(
                fn=fn,
                args=resolved_args,
                kwargs=kwargs,
                reply_to=rto,
                input_streams=stream_refs,
            ),
            timeout=timeout,
        ),
        timeout=timeout,
    )

    for t in pump_tasks:
        await t

    match result:
        case WorkerTaskSucceeded(result=_StreamHandle(producer_ref=pref)):
            source = await _resolve_output_stream(client, pref)
            loop = asyncio.get_running_loop()
            return WorkerTaskSucceeded(result=_SyncSource(source, loop), node_id=result.node_id)
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

    name = f"out-consumer-{uuid4().hex[:8]}"
    consumer_ref = client.spawn(
        stream_consumer(producer_ref, timeout=60.0, initial_demand=16),
        name,
    )
    return await client.ask(consumer_ref, lambda r: GetSource(reply_to=r), timeout=10.0)
