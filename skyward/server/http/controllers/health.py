from __future__ import annotations

from litestar import Controller, get

from skyward.server.application import ports
from skyward.shared.schemas import DependencyState, Liveness, Readiness


class HealthController(Controller):
    path = "/health"
    tags = ["health"]

    @get("/live", summary="Liveness", description="The process answers. Says nothing about the store or about providers.")
    async def live(self, health: ports.Health) -> Liveness:
        return Liveness(live=await health.live())

    @get(
        "/ready",
        summary="Readiness",
        description=(
            "Schema, command writer and initial recovery are usable. Readiness opens once every persisted record has "
            "been classified — it does not wait for new provisioning to finish."
        ),
    )
    async def ready(self, health: ports.Health) -> Readiness:
        return Readiness(ready=await health.ready())

    @get(
        "/dependencies",
        summary="Dependency health",
        description=(
            "One entry per thing the daemon leans on, keyed by its name. A provider that is unavailable degrades the "
            "computes on it; it never fails liveness, because killing a daemon over one marketplace having an afternoon "
            "would take every other compute down with it."
        ),
    )
    async def dependency_health(self, health: ports.Health) -> dict[str, DependencyState]:
        return await health.dependencies()
