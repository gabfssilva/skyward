"""Bootstrap DSL re-exported from skyward.bootstrap.

The existing bootstrap system is already well-structured.
This module simply re-exports it for v2 users.

Example:
    >>> from skyward.v2.bootstrap import bootstrap, apt, pip, checkpoint
    >>>
    >>> script = bootstrap(
    ...     apt("python3", "curl"),
    ...     pip("torch", "transformers"),
    ...     checkpoint(".ready"),
    ... )
"""

# Re-export everything from the existing bootstrap module
from skyward.bootstrap import *  # noqa: F401, F403
from skyward.bootstrap import __all__
