from __future__ import annotations

from litestar import Controller, get

from skyward.server.application import ports


class HealthController(Controller):
    path = "/health"
    tags = ["health"]

    @get("/live", summary="Liveness", description="The process answers. Says nothing about the store or about providers.")
    async def live(self, health: ports.Health) -> dict[str, bool]:
        return {"live": await health.live()}

    @get(
        "/ready",
        summary="Readiness",
        description=(
            "Schema, command writer and initial recovery are usable. Readiness opens once every persisted record has "
            "been classified — it does not wait for new provisioning to finish."
        ),
    )
    async def ready(self, health: ports.Health) -> dict[str, bool]:
        return {"ready": await health.ready()}

    @get(
        "/dependencies",
        summary="Dependency health",
        description="Providers, store and workers. An unavailable provider degrades a compute; it never fails liveness.",
    )
    async def dependency_health(self, health: ports.Health) -> dict[str, str]:
        return await health.dependencies()
