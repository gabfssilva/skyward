from __future__ import annotations

from typing import Any

from skyward.shared.schemas import ErrorCode


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


class NameTakenError(SkywardError):
    code = "name_taken"
    status = 409


class ComputeNotConnectedError(SkywardError):
    code = "compute_not_connected"
    status = 409
    retryable = True


class ComputeNotAcceptingError(SkywardError):
    code = "compute_not_accepting"
    status = 422


class ComputeNotResizableError(SkywardError):
    code = "compute_not_resizable"
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


