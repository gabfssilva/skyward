"""Panel actor - Rich terminal dashboard driven by spy events.

Panel tells this story: idle -> observing -> stopped.
"""

from __future__ import annotations

import time
from dataclasses import replace
from types import MappingProxyType

from casty import ActorContext, Behavior, Behaviors, SpyEvent, Terminated

from skyward.accelerators.catalog import get_gpu_vram_gb
from skyward.api.spec import PoolSpec
from skyward.observability.logger import logger
from skyward.observability.panel.renderer import PanelRenderer
from skyward.observability.panel.state import InfraState, InstanceState, MetricsState, PanelState

from .messages import (
    BootstrapCommand,
    BootstrapConsole,
    BootstrapPhase,
    ClusterProvisioned,
    ClusterReady,
    ClusterRequested,
    Error,
    InstanceBootstrapped,
    InstanceDestroyed,
    InstancePreempted,
    InstanceProvisioned,
    Log,
    Metric,
    ShutdownRequested,
)

type PanelInput = SpyEvent


def panel_actor(spec: PoolSpec) -> Behavior[PanelInput]:
    """Panel tells this story: idle -> observing -> stopped."""
    log = logger.bind(actor="panel")

    def idle() -> Behavior[PanelInput]:
        async def setup(ctx: ActorContext[PanelInput]) -> Behavior[PanelInput]:
            renderer = PanelRenderer()
            return observing(PanelState(), renderer)
        return Behaviors.setup(setup)

    def observing(state: PanelState, renderer: PanelRenderer) -> Behavior[PanelInput]:
        async def receive(ctx: ActorContext[PanelInput], msg: PanelInput) -> Behavior[PanelInput]:
            match msg:
                case SpyEvent(event=Terminated()):
                    return Behaviors.same()

                # ─── Cluster Lifecycle ────────────────────────────────
                case SpyEvent(event=ClusterRequested() as ev):
                    log.debug("Phase -> Provisioning, nodes={n}", n=ev.spec.nodes)
                    instances = MappingProxyType({
                        f"node-{i}": InstanceState(f"node-{i}")
                        for i in range(ev.spec.nodes)
                    })
                    gpu_model = ev.spec.accelerator_name or ""
                    gpu_count = ev.spec.accelerator_count
                    gpu_vram = get_gpu_vram_gb(gpu_model) if gpu_model else 0
                    new_state = replace(state,
                        start_time=time.monotonic(),
                        total_nodes=ev.spec.nodes,
                        phase="Provisioning",
                        is_done=False,
                        has_error=False,
                        instances=instances,
                        infra=InfraState(
                            provider=ev.provider,
                            region=ev.spec.region,
                            gpu_count=gpu_count,
                            gpu_model=gpu_model,
                            gpu_vram_gb=gpu_vram,
                            allocation=ev.spec.allocation,
                        ),
                    )
                    renderer.start(new_state)
                    return observing(new_state, renderer)

                case SpyEvent(event=ClusterProvisioned()):
                    log.debug("Phase -> Bootstrapping")
                    elapsed = time.monotonic() - state.start_time if state.start_time else 0.0
                    new_state = replace(state,
                        phase="Bootstrapping",
                        phase_times=MappingProxyType({**state.phase_times, "provision": elapsed}),
                    )
                    renderer.update_state(new_state)
                    return observing(new_state, renderer)

                case SpyEvent(event=ClusterReady(cluster=cluster)):
                    nodes = cluster.instances
                    log.debug("Phase -> Executing, ready_nodes={n}", n=len(nodes))
                    elapsed = time.monotonic() - state.start_time if state.start_time else 0.0
                    new_state = replace(state,
                        phase="Executing",
                        ready=len(nodes),
                        phase_times=MappingProxyType({**state.phase_times, "bootstrap": elapsed}),
                    )
                    renderer.update_state(new_state)
                    return observing(new_state, renderer)

                case SpyEvent(event=ShutdownRequested()):
                    log.debug("Phase -> Shutting down")
                    now = time.monotonic()
                    new_instances = {}
                    for iid, inst in state.instances.items():
                        if inst.end_time is None and inst.start_time is not None:
                            new_instances[iid] = replace(inst, end_time=now)
                        else:
                            new_instances[iid] = inst
                    new_state = replace(state,
                        is_done=True,
                        phase="Shutting down",
                        instances=MappingProxyType(new_instances),
                    )
                    renderer.update_state(new_state)
                    renderer.stop()
                    total_cost, elapsed, savings = _calculate_cost(new_state)
                    renderer.print_final_status(
                        has_error=new_state.has_error,
                        total_cost=total_cost,
                        elapsed=elapsed,
                        savings=savings,
                    )
                    return observing(new_state, renderer)

                # ─── Instance Lifecycle ───────────────────────────────
                case SpyEvent(event=InstanceProvisioned() as ev):
                    inst = ev.instance
                    placeholder = f"node-{state.provisioned}"
                    remaining = {k: v for k, v in state.instances.items() if k != placeholder}
                    remaining[inst.id] = InstanceState(
                        instance_id=inst.id,
                        node=inst.node,
                        provider=inst.provider,
                        is_spot=inst.spot,
                        hourly_rate=inst.hourly_rate,
                        on_demand_rate=inst.on_demand_rate,
                        billing_increment_minutes=inst.billing_increment,
                        spec_name=inst.instance_type,
                        start_time=time.monotonic(),
                        metrics=MetricsState(),
                    )

                    new_infra = state.infra
                    if not state.infra.instance_type:
                        new_infra = InfraState(
                            provider=inst.provider,
                            region=inst.region or spec.region,
                            instance_type=inst.instance_type,
                            vcpus=inst.vcpus,
                            memory_gb=int(inst.memory_gb),
                            gpu_count=inst.gpu_count or state.infra.gpu_count,
                            gpu_model=inst.gpu_model or state.infra.gpu_model,
                            gpu_vram_gb=inst.gpu_vram_gb or state.infra.gpu_vram_gb,
                            allocation=state.infra.allocation,
                        )

                    new_state = replace(state,
                        provisioned=state.provisioned + 1,
                        spot_count=state.spot_count + (1 if inst.spot else 0),
                        ondemand_count=state.ondemand_count + (0 if inst.spot else 1),
                        instances=MappingProxyType(remaining),
                        infra=new_infra,
                    )
                    renderer.update_state(new_state)
                    return observing(new_state, renderer)

                case SpyEvent(event=InstanceBootstrapped() as ev):
                    inst = state.instances.get(ev.instance.id)
                    new_instances = dict(state.instances)
                    if inst:
                        new_instances[ev.instance.id] = replace(inst, bootstrapped=True)
                    new_ready = state.ready + 1
                    total = state.total_nodes or state.provisioned
                    new_phase = state.phase
                    new_phase_times = dict(state.phase_times)
                    if new_ready >= total > 0:
                        new_phase = "Executing"
                        elapsed = time.monotonic() - state.start_time if state.start_time else 0.0
                        new_phase_times["bootstrap"] = elapsed
                    new_state = replace(state,
                        ready=new_ready,
                        phase=new_phase,
                        phase_times=MappingProxyType(new_phase_times),
                        instances=MappingProxyType(new_instances),
                    )
                    renderer.update_state(new_state)
                    return observing(new_state, renderer)

                case SpyEvent(event=InstanceDestroyed() as ev):
                    inst = state.instances.get(ev.instance_id)
                    if inst:
                        new_instances = {
                            **state.instances,
                            ev.instance_id: replace(inst, end_time=time.monotonic()),
                        }
                        new_state = replace(
                            state, instances=MappingProxyType(new_instances),
                        )
                        renderer.update_state(new_state)
                        return observing(new_state, renderer)
                    return Behaviors.same()

                case SpyEvent(event=InstancePreempted() as ev):
                    inst = state.instances.get(ev.instance.id)
                    if inst:
                        new_inst = replace(
                            inst,
                            preempted=True,
                            logs=(*inst.logs, f"PREEMPTED: {ev.reason}")[-100:],
                        )
                        new_instances = {**state.instances, ev.instance.id: new_inst}
                        new_state = replace(state, instances=MappingProxyType(new_instances))
                        renderer.update_state(new_state)
                        return observing(new_state, renderer)
                    return Behaviors.same()

                # ─── Bootstrap Events ─────────────────────────────────
                case SpyEvent(event=BootstrapConsole() as ev):
                    inst = state.instances.get(ev.instance.id)
                    if inst and ev.content.strip():
                        line = ev.content.strip()
                        if not line.startswith("#"):
                            new_inst = _add_log(inst, line[:80])
                            new_instances = {
                                **state.instances,
                                ev.instance.id: new_inst,
                            }
                            new_state = replace(
                                state,
                                instances=MappingProxyType(new_instances),
                            )
                            renderer.update_state(new_state)
                            return observing(new_state, renderer)
                    return Behaviors.same()

                case SpyEvent(event=BootstrapPhase() as ev):
                    inst = state.instances.get(ev.instance.id)
                    if not inst:
                        return Behaviors.same()
                    match ev.event:
                        case "started":
                            new_inst = _add_log(inst, f"$ {ev.phase}...")
                        case "completed":
                            elapsed_str = f" ({ev.elapsed:.1f}s)" if ev.elapsed else ""
                            new_inst = _add_log(inst, f"  {ev.phase} done{elapsed_str}")
                        case "failed":
                            new_inst = _add_log(inst, f"  {ev.phase} FAILED: {ev.error}")
                        case _:
                            new_inst = inst
                    new_state = replace(state,
                        phase="Bootstrapping",
                        instances=MappingProxyType({**state.instances, ev.instance.id: new_inst}),
                    )
                    renderer.update_state(new_state)
                    return observing(new_state, renderer)

                case SpyEvent(event=BootstrapCommand() as ev):
                    inst = state.instances.get(ev.instance.id)
                    if inst:
                        cmd = ev.command.strip()
                        if cmd:
                            display_cmd = f"$ {cmd[:70]}..." if len(cmd) > 70 else f"$ {cmd}"
                            new_inst = _add_log(inst, display_cmd)
                            new_instances = {
                                **state.instances,
                                ev.instance.id: new_inst,
                            }
                            new_state = replace(
                                state,
                                instances=MappingProxyType(new_instances),
                            )
                            renderer.update_state(new_state)
                            return observing(new_state, renderer)
                    return Behaviors.same()

                # ─── Metrics and Logs ─────────────────────────────────
                case SpyEvent(event=Metric() as ev):
                    inst = state.instances.get(ev.instance.id)
                    if not inst:
                        return Behaviors.same()
                    alpha = 0.15
                    prev = inst.metrics.smoothed.get(ev.name, ev.value)
                    smoothed = alpha * ev.value + (1 - alpha) * prev
                    new_history = (*inst.metrics.history.get(ev.name, ()), smoothed)[-25:]
                    new_metrics = replace(inst.metrics,
                        values=MappingProxyType({**inst.metrics.values, ev.name: smoothed}),
                        history=MappingProxyType({**inst.metrics.history, ev.name: new_history}),
                        smoothed=MappingProxyType({**inst.metrics.smoothed, ev.name: smoothed}),
                    )
                    new_inst = replace(inst, metrics=new_metrics)
                    new_instances = {
                        **state.instances,
                        ev.instance.id: new_inst,
                    }
                    new_state = replace(
                        state,
                        instances=MappingProxyType(new_instances),
                    )
                    renderer.update_state(new_state)
                    return observing(new_state, renderer)

                case SpyEvent(event=Log() as ev):
                    inst = state.instances.get(ev.instance.id)
                    if inst:
                        line = ev.line.strip()
                        if line:
                            new_inst = _add_log(inst, line)
                            new_instances = {
                                **state.instances,
                                ev.instance.id: new_inst,
                            }
                            new_state = replace(
                                state,
                                instances=MappingProxyType(new_instances),
                            )
                            renderer.update_state(new_state)
                            return observing(new_state, renderer)
                    return Behaviors.same()

                # ─── Errors ───────────────────────────────────────────
                case SpyEvent(event=Error() as ev):
                    log.debug("Error received, fatal={fatal}", fatal=ev.fatal)
                    new_state = replace(state,
                        has_error=True,
                        phase="Error" if ev.fatal else state.phase,
                    )
                    renderer.update_state(new_state)
                    return observing(new_state, renderer)

                case _:
                    return Behaviors.same()

        return Behaviors.receive(receive)

    return idle()


# ─── Pure Helpers ─────────────────────────────────────────────────


def _add_log(inst: InstanceState, line: str) -> InstanceState:
    """Add a log line to an instance, handling in-place terminal updates.

    Returns a new InstanceState with the updated logs.
    """
    clean_line = line.replace("\x08", "").replace("\r", "").replace("\n", " ")
    if not clean_line.strip():
        return inst

    is_progress = any(c in clean_line for c in "━▁▂▃▄▅▆▇█") or "/step" in clean_line

    if is_progress and inst.logs:
        last = inst.logs[-1]
        last_is_progress = any(c in last for c in "━▁▂▃▄▅▆▇█") or "/step" in last
        if last_is_progress:
            new_logs = (*inst.logs[:-1], clean_line)
            return replace(inst, logs=new_logs, last_log_time=time.monotonic())

    new_logs = (*inst.logs, clean_line)[-100:]
    return replace(inst, logs=new_logs, last_log_time=time.monotonic())


def _calculate_cost(state: PanelState) -> tuple[float, float, float]:
    """Calculate (total_cost, max_elapsed, savings)."""
    total_cost = 0.0
    total_ondemand = 0.0
    max_elapsed = 0.0

    for inst in state.instances.values():
        if inst.start_time is not None and not inst.is_placeholder:
            total_cost += inst.cost
            total_ondemand += inst.on_demand_cost
            max_elapsed = max(max_elapsed, inst.elapsed_seconds)

    savings = total_ondemand - total_cost
    return total_cost, max_elapsed, savings
