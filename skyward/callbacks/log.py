"""Log callback using loguru with custom colors."""

from __future__ import annotations

import sys
from datetime import datetime

from loguru import logger

from skyward.events import (
    BootstrapCompleted,
    BootstrapProgress,
    BootstrapStarting,
    CostFinal,
    CostUpdate,
    Error,
    InfraCreated,
    InfraCreating,
    InstanceLaunching,
    InstanceProvisioned,
    InstanceStopping,
    LogLine,
    Metrics,
    PoolStarted,
    PoolStopping,
    ProvisioningCompleted,
    RegionAutoSelected,
    SkywardEvent,
)

# Keycap number emojis (0-9)
_NODE_EMOJIS = ("0️⃣", "1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣")


def _node_emoji(node: int) -> str:
    """Convert node number to keycap emoji(s)."""
    if 0 <= node <= 9:
        return _NODE_EMOJIS[node]
    # For 10+, combine multiple digit emojis
    return "".join(_NODE_EMOJIS[int(d)] for d in str(node))


# Configure loguru with custom level (no auto-color)
logger.remove()
logger.level("OUTPUT", no=25, color="")
logger.add(sys.stderr, format="{message}", level="OUTPUT")


def _emit(msg: str) -> None:
    """Emit colored message via loguru."""
    logger.opt(colors=True).log("OUTPUT", msg)


def _timestamp() -> str:
    """Get current timestamp (dim)."""
    return f"<dim>{datetime.now().strftime('%H:%M:%S')}</dim>"


def _format_level(level: str) -> str:
    """Format level with appropriate color."""
    match level:
        case "INFO":
            return "<green>INFO   </green>"
        case "ERROR":
            return "<red>ERROR  </red>"
        case "WARNING":
            return "<yellow>WARNING</yellow>"
        case _:
            return f"{level:<7}"


def _log_local(level: str, msg: str) -> None:
    """Log with local source."""
    ts = _timestamp()
    lvl = _format_level(level)
    _emit(f"{ts} | {'orchestrator':<20} | {lvl} | <bold>{msg}</bold>")


def _log_cloud(level: str, instance_id: str, node: int, msg: str) -> None:
    """Log with cloud source (blue text)."""
    ts = _timestamp()
    source = f"{instance_id[:12]} {_node_emoji(node)}"
    lvl = _format_level(level)
    _emit(f"{ts} | <blue>{source:<20}</blue> | {lvl} | <bold>{msg}</bold>")

def _log_metrics(level: str, instance_id: str, node: int,  msg: str) -> None:
    """Log with metrics source."""
    ts = _timestamp()
    source = f"{instance_id[:12]} {_node_emoji(node)}"
    lvl = _format_level(level)
    _emit(f"{ts} | <yellow>{source:<20}</yellow> | {lvl} | <bold>{msg}</bold>")


def log(event: SkywardEvent) -> None:
    """Callback that prints formatted events to stdout with emojis.

    Metrics events are silenced by default as they are too verbose.
    Use a custom callback if you need metrics logging.

    Args:
        event: The event to log.
    """
    match event:
        # Lifecycle
        case PoolStarted():
            pass  # Silent

        case PoolStopping():
            pass  # Silent

        # Provision phase
        case InfraCreating():
            _log_local("INFO", "⚙️  Creating infrastructure...")

        case InfraCreated(region=region):
            _log_local("INFO", f"⚙️  Infrastructure ready ({region})")

        case InstanceLaunching(count=count, instance_type=itype, provider=provider):
            _log_local("INFO", f"🚀 Launching {count}x {itype} on {provider.name}")

        case InstanceProvisioned(instance_id=iid, spot=spot, instance_type=itype, provider=provider):
            tag = "[spot]" if spot else "[on-demand]"
            _log_local("INFO", f"📦 {iid[:12]} {tag} {itype} provisioned on {provider.name}")

        case ProvisioningCompleted(spot=spot, on_demand=on_demand, provider=prov, region=region, instances=instances):
            total = spot + on_demand
            inst = map(lambda i: i[:12], instances)

            if spot and on_demand:
                _log_local("INFO", f"🖥️ {total} instances provisioned ({spot} spot + {on_demand} on-demand) on {prov.value} ({region}) with ids {', '.join(inst)}")
            elif spot:
                _log_local("INFO", f"🖥️ {total} spot instances provisioned on {prov.value} ({region}) with ids {', '.join(inst)}")
            else:
                _log_local("INFO", f"🖥️ {total} instances provisioned on {prov.value} ({region}) with ids {', '.join(inst)}")

        case RegionAutoSelected(requested_region=req, selected_region=sel, instance_type=itype, provider=prov):
            _log_local("WARNING", f"⚠️  {itype} not available in {req}, using {sel} instead")

        # Setup phase
        case BootstrapStarting(instance_id=iid):
            _log_local("INFO", f"⚙️ Bootstrap starting for {iid[:12]}...")

        case BootstrapProgress(instance_id=iid, step=step):
            _log_local("INFO", f"⚙️  {iid[:12]} {step}")

        case BootstrapCompleted(instance_id=iid):
            _log_local("INFO", f"✅ {iid[:12]} is ready.")

        # Execution phase - cloud logs with instance source
        case LogLine(node=node, instance_id=iid, line=line) if line.strip():
            _log_cloud("INFO", iid, node, f"☁️  {line.rstrip()}")

        case Metrics(node=node, instance_id=iid) as m:
            msg = f"CPU {m.cpu_percent:.1f}% | Mem {m.memory_used_mb:.0f}/{m.memory_total_mb:.0f}MB"
            if m.gpu_utilization is not None:
                msg += f" | GPU {m.gpu_utilization:.0f}% {m.gpu_memory_used_mb:.0f}/{m.gpu_memory_total_mb:.0f}MB"
            _log_metrics("INFO", iid, node, msg)

        # Cost events
        case CostUpdate(
            accumulated_cost=cost,
            elapsed_seconds=elapsed,
            hourly_rate=hourly,
            spot_count=spot,
            ondemand_count=ondemand,
        ):
            mins, secs = divmod(int(elapsed), 60)
            _log_local(
                "INFO",
                f"💰 ${cost:.4f} ({mins}m{secs:02d}s) | "
                f"${hourly:.2f}/hr | {spot} spot + {ondemand} on-demand",
            )

        case CostFinal(
            total_cost=cost,
            total_seconds=elapsed,
            spot_count=spot,
            ondemand_count=ondemand,
            savings_vs_ondemand=savings,
        ):
            mins, secs = divmod(int(elapsed), 60)
            _log_local(
                "INFO",
                f"💰 Final: ${cost:.4f} ({mins}m{secs:02d}s) | "
                f"{spot} spot + {ondemand} on-demand | saved ${savings:.4f}",
            )

        # Shutdown phase
        case InstanceStopping(instance_id=iid):
            _log_local("INFO", f"⏹️  {iid[:12]} Stopping...")

        # Errors
        case Error(message=msg, instance_id=iid):
            if iid:
                _log_cloud("ERROR", iid, 0, f"❌ {msg}")
            else:
                _log_local("ERROR", f"❌ {msg}")

        case _:
            pass  # Unknown events are silently ignored
