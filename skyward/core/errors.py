"""User-facing exceptions raised by Skyward."""

from __future__ import annotations

from dataclasses import dataclass


class ProvisioningError(RuntimeError):
    """Raised when pool provisioning fails.

    Parameters
    ----------
    pool_name
        Name of the pool that failed to provision.
    reason
        Human-readable explanation of the failure.
    """

    def __init__(self, pool_name: str, reason: str) -> None:
        self.pool_name = pool_name
        self.reason = reason
        super().__init__(f"Pool '{pool_name}' provisioning failed: {reason}")


@dataclass(frozen=True, slots=True)
class _SpecSummary:
    provider: str
    accelerator: str
    allocation: str


class NoOffersError(RuntimeError):
    """Raised when no offers match across all specs.

    Parameters
    ----------
    specs
        Summary of what was requested, for user-facing diagnostics.
    """

    def __init__(self, specs: tuple[_SpecSummary, ...]) -> None:
        self.specs = specs
        providers = ", ".join(dict.fromkeys(s.provider for s in specs))
        accels = ", ".join(dict.fromkeys(s.accelerator for s in specs))
        super().__init__(
            f"No offers found for {accels} on {providers}",
        )
