"""Control flow operations for bootstrap scripts.

Conditionals, loops, variable capture, and logical operators.
"""

from __future__ import annotations

from .compose import Op, resolve

# =============================================================================
# Variable Capture
# =============================================================================


def capture(name: str, cmd: str) -> Op:
    """Capture command output into a shell variable.

    Args:
        name: Variable name to store the result.
        cmd: Command to execute and capture output from.

    Example:
        >>> capture("UV_PATH", "which uv")()
        'UV_PATH=$(which uv)'
    """
    return lambda: f"{name}=$({cmd})"


def var(name: str) -> str:
    """Reference a captured shell variable.

    This returns a string (not an Op) for use in f-strings.

    Args:
        name: Variable name to reference.

    Returns:
        Shell variable reference string.

    Example:
        >>> var("UV_PATH")
        '${UV_PATH}'
        >>> f"{var('UV_PATH')} pip install torch"
        '${UV_PATH} pip install torch'
    """
    return f"${{{name}}}"


# =============================================================================
# Conditionals
# =============================================================================


def when(condition: str, *ops: Op) -> Op:
    """Execute operations only if condition is true.

    Args:
        condition: Shell condition (evaluated with `if condition; then`).
        *ops: Operations to execute if condition is true.

    Example:
        >>> when("command -v nvidia-smi", "nvidia-smi --query")()
        'if command -v nvidia-smi; then\\n    nvidia-smi --query\\nfi'
    """
    if not ops:
        return lambda: f"# when({condition!r}): no operations"

    def generate() -> str:
        body = "\n    ".join(resolve(op) for op in ops)
        return f"if {condition}; then\n    {body}\nfi"

    return generate


def unless(condition: str, *ops: Op) -> Op:
    """Execute operations only if condition is false.

    Args:
        condition: Shell condition to negate.
        *ops: Operations to execute if condition is false.

    Example:
        >>> unless("command -v uv", "curl ... | sh")()
        'if ! (command -v uv); then\\n    curl ... | sh\\nfi'
    """
    return when(f"! ({condition})", *ops)


# =============================================================================
# Loops
# =============================================================================


def for_each(var_name: str, items: str, *ops: Op) -> Op:
    """Execute operations for each item in a list.

    Args:
        var_name: Loop variable name.
        items: Shell expression for items (e.g., "$(seq 0 3)", "a b c").
        *ops: Operations to execute for each item.

    Example:
        >>> for_each("i", "$(seq 0 3)", "echo $i")()
        'for i in $(seq 0 3); do\\n    echo $i\\ndone'
    """
    if not ops:
        return lambda: f"# for_each({var_name!r}, {items!r}): no operations"

    def generate() -> str:
        body = "\n    ".join(resolve(op) for op in ops)
        return f"for {var_name} in {items}; do\n    {body}\ndone"

    return generate


# =============================================================================
# Logical Operators
# =============================================================================


def and_then(*cmds: str) -> str:
    """Chain commands with && (all must succeed).

    This returns a string (not an Op) for direct use.

    Args:
        *cmds: Commands to chain.

    Returns:
        Commands joined with ' && '.

    Example:
        >>> and_then("mkdir /tmp/test", "cd /tmp/test", "touch file")
        'mkdir /tmp/test && cd /tmp/test && touch file'
    """
    return " && ".join(cmds)


def or_else(*cmds: str) -> str:
    """Chain commands with || (try alternatives on failure).

    This returns a string (not an Op) for direct use.

    Args:
        *cmds: Commands to try in order.

    Returns:
        Commands joined with ' || '.

    Example:
        >>> or_else("which uv", "curl ... | sh")
        'which uv || curl ... | sh'
    """
    return " || ".join(cmds)


# =============================================================================
# Grouping
# =============================================================================


def group(*ops: Op) -> Op:
    """Group multiple operations into one.

    Useful for passing multiple operations where one is expected.

    Args:
        *ops: Operations to group.

    Example:
        >>> group("echo a", "echo b")()
        'echo a\\necho b'
    """
    if not ops:
        return lambda: "# empty group"

    def generate() -> str:
        return "\n".join(resolve(op) for op in ops)

    return generate


def subshell(*ops: Op) -> Op:
    """Execute operations in a subshell.

    Changes to environment/directory are isolated.

    Args:
        *ops: Operations to execute in subshell.

    Example:
        >>> subshell("cd /tmp", "touch file")()
        '(\\n    cd /tmp\\n    touch file\\n)'
    """
    if not ops:
        return lambda: "# empty subshell"

    def generate() -> str:
        body = "\n    ".join(resolve(op) for op in ops)
        return f"(\n    {body}\n)"

    return generate
