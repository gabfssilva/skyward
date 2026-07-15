from __future__ import annotations

from typing import Any

from skyward.protocol.schemas import ErrorCode


class SkywardError(Exception):
    code: ErrorCode
    status: int
    retryable: bool = False

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


class NotFoundError(SkywardError):
    code = "not_found"
    status = 404


class RevisionConflictError(SkywardError):
    code = "revision_conflict"
    status = 412
    retryable = True


class IdempotencyConflictError(SkywardError):
    code = "idempotency_conflict"
    status = 409


class LeaseHeldError(SkywardError):
    code = "lease_held"
    status = 409
    retryable = True


class ComputeNotAcceptingError(SkywardError):
    code = "compute_not_accepting"
    status = 422


class UnsupportedProviderError(SkywardError):
    code = "unsupported_provider"
    status = 422


class UnsupportedPluginError(SkywardError):
    code = "unsupported_plugin"
    status = 422


class SecretInDefinitionError(SkywardError):
    code = "secret_in_definition"
    status = 422


class HashMismatchError(SkywardError):
    code = "hash_mismatch"
    status = 400


class TaskFailedError(SkywardError):
    code = "task_failed"
    status = 409


class TaskIndeterminateError(SkywardError):
    code = "task_indeterminate"
    status = 409


class DuplicationNotAcknowledgedError(SkywardError):
    code = "duplication_not_acknowledged"
    status = 409


class CapabilityMismatchError(SkywardError):
    code = "capability_mismatch"
    status = 422


class ReleasePendingError(SkywardError):
    """The provider cannot give a resource back yet, but will be able to soon.

    A security group whose instances were terminated seconds ago still holds their
    network interfaces for a moment; the teardown is not done, it is not stuck, and
    the next pass is what finishes it — so the compute stays ``deleting``, not
    ``degraded``.
    """

    code = "release_pending"
    status = 409
    retryable = True
