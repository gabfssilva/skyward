"""Bootstrap logic for AWS EC2 instances.

Uses the common bootstrap module with AWS-specific extensions:
- SSM for remote command execution
- Extra checkpoints for S3 download and volumes
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from botocore.exceptions import ClientError

from skyward.events import EventCallback
from skyward.exceptions import InstanceTerminatedError
from skyward.providers.common.bootstrap import (
    Checkpoint,
)
from skyward.providers.common.bootstrap import (
    wait_for_bootstrap as _wait_for_bootstrap,
)

if TYPE_CHECKING:
    from skyward.providers.aws.ssm import SSMSession

logger = logging.getLogger("skyward.aws.bootstrap")


# AWS-specific checkpoints (in addition to common ones)
AWS_EXTRA_CHECKPOINTS = (
    Checkpoint(".step_download", "downloading deps"),
    Checkpoint(".step_volumes", "volumes"),
)


def _is_instance_terminated_error(exc: BaseException) -> bool:
    """Check if exception indicates the instance is terminated."""
    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code", "")
        return code in ("InvalidInstanceId", "InvalidInstanceId.NotFound")
    msg = str(exc).lower()
    return "invalidinstanceid" in msg or "not in a valid state" in msg


def _create_ssm_runner(
    ssm_session: SSMSession,
    instance_id: str,
) -> Callable[[str], str]:
    """Create a command runner using SSM.

    Returns a function that runs commands via SSM and returns stdout.
    Raises InstanceTerminatedError if instance is not available.
    """

    def run_command(cmd: str) -> str:
        try:
            result = ssm_session.run_command(instance_id, cmd, timeout=30)
            if result.success:
                return result.stdout
            # Command failed but instance is reachable
            return ""
        except Exception as e:
            if _is_instance_terminated_error(e):
                raise InstanceTerminatedError(
                    instance_id, "instance not in valid state"
                ) from e
            raise

    return run_command


def wait_for_bootstrap(
    ssm_session: SSMSession,
    instance_id: str,
    on_event: EventCallback = None,
    verified_instances: set[str] | None = None,
    timeout: int = 300,
) -> None:
    """Wait for instance bootstrap with progress tracking.

    AWS-specific wrapper that uses SSM for command execution.

    Args:
        ssm_session: SSM session for running commands.
        instance_id: EC2 instance ID.
        on_event: Optional callback for progress events.
        verified_instances: Set to track verified instances.
        timeout: Maximum time to wait in seconds.

    Raises:
        InstanceTerminatedError: If instance was terminated.
        RuntimeError: If bootstrap times out.
    """
    if verified_instances is None:
        verified_instances = set()

    runner = _create_ssm_runner(ssm_session, instance_id)

    try:
        _wait_for_bootstrap(
            run_command=runner,
            instance_id=instance_id,
            on_event=on_event,
            timeout=timeout,
            extra_checkpoints=AWS_EXTRA_CHECKPOINTS,
        )
        verified_instances.add(instance_id)
    except RuntimeError:
        # Bootstrap failed - get additional debug info
        error_file = ""
        bootstrap_log = ""
        cloud_init_logs = ""

        try:
            # Read .error file (written by trap on failure)
            result = ssm_session.run_command(
                instance_id,
                "cat /opt/skyward/.error 2>/dev/null || echo ''",
                timeout=30,
            )
            error_file = result.stdout.strip() if result.success else ""

            # Read bootstrap.log (actual script output)
            result = ssm_session.run_command(
                instance_id,
                "tail -100 /opt/skyward/bootstrap.log 2>/dev/null || echo ''",
                timeout=30,
            )
            bootstrap_log = result.stdout.strip() if result.success else ""

            # Read cloud-init logs
            result = ssm_session.run_command(
                instance_id,
                "cat /var/log/cloud-init-output.log 2>/dev/null || echo ''",
                timeout=30,
            )
            cloud_init_logs = result.stdout.strip() if result.success else ""
        except Exception:
            pass

        # Build comprehensive error message
        msg_parts = [f"Bootstrap failed on {instance_id}."]

        if error_file:
            msg_parts.append(f"\n--- Error file ---\n{error_file}")

        if bootstrap_log:
            msg_parts.append(f"\n--- Bootstrap log (last 100 lines) ---\n{bootstrap_log}")

        if cloud_init_logs and not error_file and not bootstrap_log:
            # Only show cloud-init if we don't have better logs
            msg_parts.append(f"\n--- Cloud-init logs ---\n{cloud_init_logs}")

        raise RuntimeError("\n".join(msg_parts)) from None
