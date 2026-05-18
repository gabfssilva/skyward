"""Log console actor — plain line-based output for non-TTY environments.

Renders an opinionated, grep-friendly projection of the session to
``stderr``:

- Lifecycle events (provisioning, ssh, bootstrap phase, ready, stopped).
- Failures (lost, preempted, ssh/worker/bootstrap failures, fatal errors).
- Scaling transitions (desired changed, spawning, draining).
- Remote logs (``Log.Emitted``) and ``LocalOutput`` lines.
- A periodic metrics snapshot (default 30s, override with the
  ``SKYWARD_LOG_METRICS_INTERVAL`` environment variable).

Each line is ``HH:MM:SS  <label>  <message>``.  No colors, no cursor
tricks, no buffering — ideal for CI logs, ``journalctl``, ``docker``
without ``-t`` and ``python script.py > run.log``.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from dataclasses import dataclass, field, replace
from statistics import mean
from typing import Any, cast

from casty import ActorContext, Behavior, Behaviors

from skyward.api.events import Error, Node, Pool, Scaling
from skyward.api.projection import _throughput
from skyward.api.views import PoolView, SessionView

from .messages import ConsoleInput, EventReceived, LocalOutput, LogReceived, ViewUpdated

__all__ = ["log_console_actor"]


_DEFAULT_INTERVAL = 30.0
_INSTANCE_ID_WIDTH = 8
_LABEL_WIDTH = 18


@dataclass(frozen=True, slots=True)
class _Tick:
    pass


type _LogMsg = ConsoleInput | _Tick


# ── Helpers ──────────────────────────────────────────────────────


def _metrics_interval() -> float:
    raw = os.environ.get("SKYWARD_LOG_METRICS_INTERVAL")
    if not raw:
        return _DEFAULT_INTERVAL
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_INTERVAL
    return value if value > 0 else _DEFAULT_INTERVAL


def _first_pool(view: SessionView) -> PoolView | None:
    if view.pools:
        return next(iter(view.pools.values()))
    return None


def _provider_name(pool: PoolView, inst: Any) -> str:
    if (
        pool.cluster is not None
        and getattr(pool.cluster, "spec", None) is not None
        and (name := pool.cluster.spec.provider)
    ):
        return name
    module = type(inst.offer.specific).__module__
    parts = module.split(".")
    if "providers" in parts:
        idx = parts.index("providers")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "?"


def _node_label(view: SessionView, node_id: int) -> str:
    pool = _first_pool(view)
    if pool is not None:
        node = pool.nodes.get(node_id)
        if node is not None and node.instance is not None and (iid := node.instance.id):
            return f"{iid[:_INSTANCE_ID_WIDTH]}/{node_id}"
    return f"node-{node_id}"


def _timestamp() -> str:
    return time.strftime("%H:%M:%S", time.localtime())


def _emit(label: str, message: str) -> None:
    print(
        f"{_timestamp()}  {label:<{_LABEL_WIDTH}}  {message}",
        file=sys.stderr,
        flush=True,
    )


def _fmt_count(n: float) -> str:
    return str(int(n)) if float(n).is_integer() else f"{n:g}"


def _cluster_lines(pool: PoolView | None) -> list[str]:
    """Multi-line cluster description emitted once per pool provision.

    Layout (each line is printed under the ``pool`` label):

    - ``spec: <N>× <accelerator> <memory> · <vcpus> vCPU · <memory_gb> GB``
    - ``cluster: <provider> · <region> · <instance_type> · spot|on-demand``
      followed by ``$<price>/hr × <n> = Σ $<total>/hr`` when known.
    - ``autoscale: <min>..<max> nodes`` when the pool is elastic.
    """
    if pool is None or not pool.instances:
        return []
    inst = pool.instances[0]
    it = inst.offer.instance_type
    n = len(pool.instances)

    spec_bits: list[str] = []
    if it.accelerator is not None:
        acc = it.accelerator
        mem = f" {acc.memory}" if acc.memory else ""
        spec_bits.append(f"{_fmt_count(acc.count)}× {acc.name}{mem}")
    spec_bits.append(f"{int(it.vcpus)} vCPU")
    spec_bits.append(f"{int(it.memory_gb)} GB")

    cluster_bits: list[str] = [
        _provider_name(pool, inst),
        inst.region or "?",
        it.name,
        "spot" if inst.spot else "on-demand",
    ]
    per = inst.offer.spot_price if inst.spot else inst.offer.on_demand_price
    if per:
        total = per * n
        cluster_bits.append(f"${per:.2f}/hr × {n} = \u03a3 ${total:.2f}/hr")

    lines = [
        "spec: " + " · ".join(spec_bits),
        f"cluster ({n}× nodes): " + " · ".join(cluster_bits),
    ]
    s = pool.scaling
    if s.is_elastic and s.min_nodes is not None and s.max_nodes is not None:
        lines.append(f"autoscale: {s.min_nodes}..{s.max_nodes} nodes")
    if pool.spec is not None and pool.spec.image is not None:
        image = pool.spec.image
        image_bits: list[str] = []
        python = getattr(image, "python_version", None)
        if python:
            image_bits.append(f"python {python}")
        pip = getattr(image, "pip", None) or ()
        if pip:
            image_bits.append(f"{len(pip)} pip pkg{'s' if len(pip) != 1 else ''}")
        apt = getattr(image, "apt", None) or ()
        if apt:
            image_bits.append(f"{len(apt)} apt pkg{'s' if len(apt) != 1 else ''}")
        volumes = pool.spec.volumes
        if volumes:
            image_bits.append(f"{len(volumes)} volume{'s' if len(volumes) != 1 else ''}")
        if image_bits:
            lines.append("image: " + " · ".join(image_bits))
    return lines


def _hourly(pool: PoolView) -> float:
    return sum(
        (i.offer.spot_price if i.spot else i.offer.on_demand_price) or 0.0
        for i in pool.instances
    )


def _avg_metric(pool: PoolView, name: str) -> float | None:
    vals = [n.metrics[name] for n in pool.nodes.values() if name in n.metrics]
    return mean(vals) if vals else None


def _avg_prefix(pool: PoolView, prefix: str) -> float | None:
    vals: list[float] = []
    for n in pool.nodes.values():
        for name, value in n.metrics.items():
            if name == prefix or name.startswith(f"{prefix}_"):
                vals.append(value)
    return mean(vals) if vals else None


def _metrics_line(pool: PoolView) -> str | None:
    t = pool.tasks
    has_tasks = bool(t.done or t.failed or t.running or t.queued)
    parts: list[str] = []
    if (v := _avg_prefix(pool, "gpu_util")) is not None:
        parts.append(f"gpu {v:.0f}%")
    if (v := _avg_prefix(pool, "gpu_mem_mb")) is not None:
        parts.append(f"vram {v / 1024:.1f}GB")
    if (v := _avg_metric(pool, "cpu")) is not None:
        parts.append(f"cpu {v:.0f}%")
    if (v := _avg_metric(pool, "mem")) is not None:
        parts.append(f"mem {v:.0f}%")
    if has_tasks:
        in_flight = t.queued + t.running
        parts.append(f"tasks {in_flight}/{t.done} \u2713")
        if t.failed:
            parts.append(f"{t.failed} \u2717")
        if (tp := _throughput(t)) > 0:
            parts.append(f"{tp:.1f} t/min")
    if not parts:
        return None
    if (hourly := _hourly(pool)) > 0 and pool.started_at:
        elapsed_h = (time.monotonic() - pool.started_at) / 3600
        total = hourly * elapsed_h
        parts.append(f"\u03a3 ${total:.2f}")
    return " · ".join(parts)


# ── Actor state ──────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _State:
    view: SessionView = field(default_factory=SessionView)
    started: bool = False
    cluster_logged: bool = False
    connected: frozenset[int] = field(default_factory=frozenset)
    ready: frozenset[int] = field(default_factory=frozenset)
    bootstrap_phases: frozenset[tuple[int, str]] = field(default_factory=frozenset)
    pool_ready: bool = False
    pool_stopped: bool = False


# ── Actor ────────────────────────────────────────────────────────


def log_console_actor() -> Behavior[ConsoleInput]:
    """Log console tells this story: event → one line → flush."""

    interval = _metrics_interval()
    fallback_started_at = time.monotonic()

    async def _sleep_tick() -> _Tick:
        await asyncio.sleep(interval)
        return _Tick()

    def _schedule_tick(ctx: ActorContext[_LogMsg]) -> None:
        ctx.pipe_to_self(
            _sleep_tick(),
            mapper=lambda t: t,
            on_failure=lambda _: _Tick(),
        )

    def _handle_event(state: _State, event: Any) -> _State:
        view = state.view
        match event:
            case Pool.Provisioning(total_nodes=n) if not state.started:
                _emit("pool", f"provisioning {n} nodes")
                return replace(state, started=True)

            case Pool.Provisioned(instances=instances) if (
                instances and not state.cluster_logged
            ):
                for line in _cluster_lines(_first_pool(view)):
                    _emit("pool", line)
                return replace(state, cluster_logged=True)

            case Pool.ProvisionFailed(reason=reason):
                _emit("pool", f"provision failed: {reason}")
                return state

            case Pool.NoOffers():
                _emit("pool", "no matching offers found")
                return state

            case Pool.Stopped() if not state.pool_stopped:
                pool = _first_pool(view)
                tail = ""
                if pool is not None:
                    elapsed = time.monotonic() - (pool.started_at or fallback_started_at)
                    tail = f" · elapsed {elapsed:.0f}s"
                    if pool.tasks.done or pool.tasks.failed:
                        tail += f" · {pool.tasks.done} \u2713"
                        if pool.tasks.failed:
                            tail += f" · {pool.tasks.failed} \u2717"
                    if (hourly := _hourly(pool)) > 0:
                        total = hourly * (elapsed / 3600)
                        tail += f" · \u03a3 ${total:.2f}"
                _emit("pool", f"stopped{tail}")
                return replace(state, pool_stopped=True)

            case Node.Connected(node_id=nid) if nid not in state.connected:
                _emit(_node_label(view, nid), "ssh connected")
                return replace(state, connected=state.connected | {nid})

            case Node.Ready(node_id=nid) if nid not in state.ready:
                _emit(_node_label(view, nid), "ready")
                new_ready = state.ready | {nid}
                new_state = replace(state, ready=new_ready)
                pool = _first_pool(view)
                if (
                    pool is not None
                    and not state.pool_ready
                    and pool.total_nodes > 0
                    and len(new_ready) >= pool.total_nodes
                ):
                    rate = _hourly(pool)
                    tail = f" · ${rate:.2f}/hr" if rate > 0 else ""
                    _emit("pool", f"ready · {len(new_ready)}/{pool.total_nodes} nodes{tail}")
                    new_state = replace(new_state, pool_ready=True)
                return new_state

            case Node.Lost(node_id=nid, reason=reason):
                _emit(_node_label(view, nid), f"lost: {reason}")
                return replace(
                    state,
                    ready=state.ready - {nid},
                    connected=state.connected - {nid},
                )

            case Node.Preempted(reason=reason):
                _emit("pool", f"node preempted: {reason}")
                return state

            case Node.ConnectionFailed(error=error):
                _emit("pool", f"ssh failed: {error}")
                return state

            case Node.WorkerFailed(error=error):
                _emit("pool", f"worker failed: {error}")
                return state

            case Node.Bootstrap.Started(node_id=nid, phase=phase):
                key = (nid, phase)
                if key in state.bootstrap_phases:
                    return state
                _emit(_node_label(view, nid), f"bootstrap: {phase}")
                return replace(state, bootstrap_phases=state.bootstrap_phases | {key})

            case Node.Bootstrap.Command(node_id=nid, command=command):
                _emit(_node_label(view, nid), f"$ {command}")
                return state

            case Node.Bootstrap.Output(node_id=nid, output=output, overwrite=False):
                pool = _first_pool(view)
                node = pool.nodes.get(nid) if pool is not None else None
                if node is None or node.bootstrap is None:
                    # Post-Ready: projection re-dispatches this as
                    # Log.Emitted, which LogReceived handles — avoid dup.
                    return state
                for chunk in output.splitlines():
                    if stripped := chunk.rstrip():
                        _emit(_node_label(view, nid), stripped)
                return state

            case Node.Bootstrap.Failed(node_id=nid, phase=phase, error=error):
                _emit(_node_label(view, nid), f"bootstrap failed at {phase}: {error}")
                return state

            case Scaling.DesiredChanged(desired=desired, reason=reason):
                _emit("pool", f"desired: {desired} ({reason})")
                return state

            case Scaling.Spawning(count=count):
                _emit("pool", f"spawning {count} nodes")
                return state

            case Scaling.Draining(count=count):
                _emit("pool", f"draining {count} nodes")
                return state

            case Error.Occurred(message=message, fatal=True):
                _emit("pool", f"fatal: {message}")
                return state

            case Error.Occurred(message=message):
                _emit("pool", f"error: {message}")
                return state

            case _:
                return state

    def active(state: _State) -> Behavior[_LogMsg]:
        async def receive(
            ctx: ActorContext[_LogMsg], msg: _LogMsg,
        ) -> Behavior[_LogMsg]:
            match msg:
                case ViewUpdated(view=view):
                    return active(replace(state, view=view))

                case EventReceived(event=event):
                    return active(_handle_event(state, event))

                case LogReceived(log=log):
                    _emit(_node_label(state.view, log.node_id), log.message)
                    return Behaviors.same()

                case LocalOutput(line=line):
                    if stripped := line.rstrip():
                        _emit("local", stripped)
                    return Behaviors.same()

                case _Tick():
                    pool = _first_pool(state.view)
                    if pool is not None and (line := _metrics_line(pool)) is not None:
                        _emit("pool", line)
                    _schedule_tick(ctx)
                    return Behaviors.same()

        return Behaviors.receive(receive)

    async def setup(ctx: ActorContext[_LogMsg]) -> Behavior[_LogMsg]:
        _schedule_tick(ctx)
        return active(_State())

    return cast(Behavior[ConsoleInput], Behaviors.setup(setup))
