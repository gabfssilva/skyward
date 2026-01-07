"""Bootstrap script composition.

Core types and composition functions for the declarative bootstrap DSL.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

from skyward.core.constants import SKYWARD_DIR

# =============================================================================
# Core Types
# =============================================================================

type Op = str | Callable[[], str] | list[Op]
"""Operation type: either a literal string or a function returning a string."""

def resolve(op: Op) -> str:
    """Resolve an operation to its string representation."""
    match op:
        case str(op):
            return op
        case list(op):
            return '\n'.join(map(lambda o: resolve(o), op))
        case None:
            return ""
        case _:
            return op()

# =============================================================================
# Header Template
# =============================================================================

DEFAULT_HEADER: Final = f"""#!/bin/bash
set -e

mkdir -p {SKYWARD_DIR}

exec > {SKYWARD_DIR}/bootstrap.log 2>&1

trap 'EC=$?; echo "Command failed: $BASH_COMMAND" > {SKYWARD_DIR}/.error; echo "Exit code: $EC" >> {SKYWARD_DIR}/.error; echo "--- Output ---" >> {SKYWARD_DIR}/.error; tail -50 {SKYWARD_DIR}/bootstrap.log >> {SKYWARD_DIR}/.error' ERR

export DEBIAN_FRONTEND=noninteractive
export PATH="/root/.local/bin:$PATH"

"""


# =============================================================================
# Composition
# =============================================================================


def bootstrap(*ops: Op | None, header: str | None = None) -> str:
    """Compose operations into a complete bootstrap script.

    Args:
        *ops: Operations to compose. Can be strings or callables returning strings.
        header: Optional custom header. Defaults to DEFAULT_HEADER.

    Returns:
        Complete shell script string.

    Example:
        >>> script = bootstrap(
        ...     apt("python3", "curl"),
        ...     checkpoint(".step_apt"),
        ...     "echo 'custom command'",  # string literals work too
        ... )
    """
    base = header if header is not None else DEFAULT_HEADER
    commands = [resolve(op) for op in ops]
    return base + "\n\n".join(commands)
