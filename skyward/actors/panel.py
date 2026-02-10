from __future__ import annotations

from typing import TYPE_CHECKING

from casty import ActorContext, Behavior, Behaviors, SpyEvent, Terminated

from skyward.messages import (
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
from skyward.spec import PoolSpec

if TYPE_CHECKING:
    from skyward.observability.panel.component import PanelComponent

type PanelMsg = (
    ClusterRequested
    | ClusterProvisioned
    | ClusterReady
    | ShutdownRequested
    | InstanceProvisioned
    | InstanceBootstrapped
    | InstanceDestroyed
    | InstancePreempted
    | BootstrapConsole
    | BootstrapPhase
    | BootstrapCommand
    | Metric
    | Log
    | Error
)

type PanelInput = SpyEvent


def panel_actor(spec: PoolSpec) -> Behavior[PanelInput]:
    """Panel actor: idle -> observing -> stopped.

    Read-only observer that drives the Rich terminal dashboard.
    Receives SpyEvent wrappers and unwraps them to update PanelState.
    """

    def idle() -> Behavior[PanelInput]:
        async def setup(ctx: ActorContext[PanelInput]) -> Behavior[PanelInput]:
            from skyward.observability.panel.component import PanelComponent
            from skyward.observability.panel.renderer import PanelRenderer
            from skyward.observability.panel.state import PanelState

            component = PanelComponent.__new__(PanelComponent)
            component.spec = spec
            component._active = False
            component._state = PanelState()
            component._renderer = PanelRenderer()

            return observing(component)

        return Behaviors.setup(setup)

    def observing(component: PanelComponent) -> Behavior[PanelInput]:
        async def receive(ctx: ActorContext[PanelInput], msg: PanelInput) -> Behavior[PanelInput]:
            match msg:
                case SpyEvent(event=Terminated()):
                    return Behaviors.same()
                case SpyEvent(event=event):
                    await _handle(component, event)
            return Behaviors.same()

        return Behaviors.receive(receive)

    return idle()


async def _handle(component: PanelComponent, event: object) -> None:
    match event:
        case ClusterRequested():
            await component._on_cluster_requested(None, event)
        case ClusterProvisioned():
            await component._on_cluster_provisioned(None, event)
        case ClusterReady():
            await component._on_cluster_ready(None, event)
        case ShutdownRequested():
            await component._on_shutdown(None, event)
        case InstanceProvisioned():
            await component._on_instance_provisioned(None, event)
        case InstanceBootstrapped():
            await component._on_instance_bootstrapped(None, event)
        case InstanceDestroyed():
            await component._on_instance_destroyed(None, event)
        case InstancePreempted():
            await component._on_preempted(None, event)
        case BootstrapConsole():
            await component._on_bootstrap_console(None, event)
        case BootstrapPhase():
            await component._on_bootstrap_phase(None, event)
        case BootstrapCommand():
            await component._on_bootstrap_command(None, event)
        case Metric():
            await component._on_metric(None, event)
        case Log():
            await component._on_log(None, event)
        case Error():
            await component._on_error(None, event)


__all__ = ["PanelMsg", "PanelInput", "panel_actor"]
