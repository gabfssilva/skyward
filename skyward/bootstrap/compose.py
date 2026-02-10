"""Bootstrap script composition.

Core types and composition functions for the declarative bootstrap DSL.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Final

from ..constants import SKYWARD_DIR

EMIT_SH_PATH: Final = f"{SKYWARD_DIR}/emit.sh"

if TYPE_CHECKING:
    from ..metrics import Metric

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
            return "\n".join(map(lambda o: resolve(o), op))
        case None:
            return ""
        case _:
            return op()


# =============================================================================
# Header Template (without metrics - those are generated dynamically)
# =============================================================================

HEADER_TEMPLATE: Final = f"""#!/bin/bash
set -e

mkdir -p {SKYWARD_DIR}

# Clear stale data from previous runs (VastAI containers can inherit state)
rm -f {SKYWARD_DIR}/events.jsonl {SKYWARD_DIR}/events.jsonl.1 {SKYWARD_DIR}/pyproject.toml {SKYWARD_DIR}/uv.lock

export DEBIAN_FRONTEND=noninteractive
export PATH="/root/.local/bin:$PATH"

# =============================================================================
# Events JSONL Streaming
# =============================================================================

EVENTS_LOG="{SKYWARD_DIR}/events.jsonl"
MAX_SIZE=10485760  # 10MB rotation threshold

# Base emit with automatic rotation
emit() {{{{
    echo "$1" >> "$EVENTS_LOG"
    local size=$(stat -c%s "$EVENTS_LOG" 2>/dev/null || stat -f%z "$EVENTS_LOG" 2>/dev/null || echo 0)
    if [ "$size" -gt "$MAX_SIZE" ]; then
        mv "$EVENTS_LOG" "${{{{EVENTS_LOG}}}}.1"
    fi
}}}}

emit_phase() {{{{
    local event="$1" phase="$2" elapsed="${{{{3:-null}}}}" error="${{{{4:-null}}}}"
    emit "{{{{\\\"type\\\":\\\"phase\\\",\\\"event\\\":\\\"$event\\\",\\\"phase\\\":\\\"$phase\\\",\\\"elapsed\\\":$elapsed,\\\"error\\\":$error}}}}"
}}}}

emit_console() {{{{
    local content="$1" stream="${{{{2:-stdout}}}}"
    # Escape for JSON: backslash first, then quotes, then control chars
    content="${{{{content//\\\\/\\\\\\\\}}}}"
    content="${{{{content//\\"/\\\\\\"}}}}"
    content="${{{{content//$'\\n'/\\\\n}}}}"
    content="${{{{content//$'\\t'/\\\\t}}}}"
    content="${{{{content//$'\\r'/\\\\r}}}}"
    emit "{{{{\\\"type\\\":\\\"console\\\",\\\"content\\\":\\\"$content\\\",\\\"stream\\\":\\\"$stream\\\"}}}}"
}}}}

emit_command() {{{{
    local cmd="$1"
    # Escape for JSON: backslash first, then quotes, then control chars
    cmd="${{{{cmd//\\\\/\\\\\\\\}}}}"
    cmd="${{{{cmd//\\"/\\\\\\"}}}}"
    cmd="${{{{cmd//$'\\n'/\\\\n}}}}"
    cmd="${{{{cmd//$'\\t'/\\\\t}}}}"
    cmd="${{{{cmd//$'\\r'/\\\\r}}}}"
    emit "{{{{\\\"type\\\":\\\"command\\\",\\\"command\\\":\\\"$cmd\\\"}}}}"
}}}}

# Save emit helpers for post-bootstrap reuse (e.g., Casty log streaming)
(echo 'EVENTS_LOG="{SKYWARD_DIR}/events.jsonl"'
 echo 'MAX_SIZE=10485760'
 declare -f emit emit_phase emit_console emit_command) > {SKYWARD_DIR}/emit.sh

run_phase() {{{{
    local phase="$1"; shift
    emit_phase "started" "$phase"

    # Emit command - extract real command if bash -c '...'
    if [ "$1" = "bash" ] && [ "$2" = "-c" ]; then
        emit_command "$3"
    else
        emit_command "$*"
    fi

    local start_ns=$(date +%s%N)

    set +e
    "$@" 2>&1 | while IFS= read -r line; do
        emit_console "$line"
    done
    local exit_code=${{{{PIPESTATUS[0]}}}}
    set -e

    local end_ns=$(date +%s%N)
    local elapsed=$(awk "BEGIN {{{{printf \\\"%.2f\\\", ($end_ns - $start_ns) / 1000000000}}}}")

    if [ $exit_code -eq 0 ]; then
        emit_phase "completed" "$phase" "$elapsed"
    else
        emit_phase "failed" "$phase" "$elapsed" "\\\"exit_code=$exit_code\\\""
        exit $exit_code
    fi
}}}}

{{metrics_script}}

# Error trap - emit failure event
trap 'emit_phase "failed" "bootstrap" null "\\\"$BASH_COMMAND failed\\\""' ERR

# Start bootstrap
emit_phase "started" "bootstrap"

"""


# =============================================================================
# Metrics Script Generation
# =============================================================================


def generate_metrics_script(metrics: tuple[Metric, ...] | list[Metric] | None) -> str:
    """Generate bash script for dynamic metrics collection.

    Each metric runs as a background process with its own collection interval.
    Multi-value metrics (e.g., GPU with index=None) emit separate events for
    each line of output (gpu_util_0, gpu_util_1, etc.).

    Args:
        metrics: Tuple of Metric definitions. If None or empty, returns no-op functions.

    Returns:
        Bash script fragment with collector functions and start/stop daemon functions.
    """
    if not metrics:
        return """# =============================================================================
# Metrics Collection (disabled)
# =============================================================================

start_metrics_daemon() { :; }
stop_metrics_daemon() { :; }
"""

    lines = [
        "# =============================================================================",
        "# Metrics Collection",
        "# =============================================================================",
        "",
    ]

    # Generate collector function for each metric
    for m in metrics:
        collector = _generate_collector(m)
        lines.append(collector)

    # Generate start_metrics_daemon
    lines.append("start_metrics_daemon() {")
    for m in metrics:
        safe_name = _safe_function_name(m.name)
        lines.append(f"    _collect_{safe_name} &")
        lines.append(f"    echo $! >> {SKYWARD_DIR}/metrics.pids")
    lines.append("}")
    lines.append("")

    # Generate stop_metrics_daemon
    lines.append(f"""stop_metrics_daemon() {{
    if [ -f {SKYWARD_DIR}/metrics.pids ]; then
        while read pid; do
            kill "$pid" 2>/dev/null || true
        done < {SKYWARD_DIR}/metrics.pids
        rm -f {SKYWARD_DIR}/metrics.pids
    fi
}}""")

    return "\n".join(lines)


def _safe_function_name(name: str) -> str:
    """Convert metric name to safe bash function name."""
    return name.replace("-", "_").replace(".", "_")


def _generate_collector(m: Metric) -> str:
    """Generate a collector function for a single metric.

    For multi-value metrics, the command output is iterated line by line,
    emitting name_0, name_1, etc. for each line.
    """
    safe_name = _safe_function_name(m.name)

    if m.multi:
        # Multi-value: each line becomes name_0, name_1, etc.
        return f"""_collect_{safe_name}() {{
    while true; do
        local ts=$(date +%s.%N)
        local idx=0
        {m.command} | while read val; do
            if [ -n "$val" ]; then
                emit "{{\\"type\\":\\"metric\\",\\"name\\":\\"{m.name}_$idx\\",\\"value\\":$val,\\"ts\\":$ts}}"
            fi
            idx=$((idx + 1))
        done
        sleep {m.interval}
    done
}}
"""
    else:
        # Single value metric
        return f"""_collect_{safe_name}() {{
    while true; do
        local ts=$(date +%s.%N)
        local val=$({m.command})
        if [ -n "$val" ]; then
            emit "{{\\"type\\":\\"metric\\",\\"name\\":\\"{m.name}\\",\\"value\\":$val,\\"ts\\":$ts}}"
        fi
        sleep {m.interval}
    done
}}
"""


# =============================================================================
# Composition
# =============================================================================


def make_header(metrics: tuple[Metric, ...] | list[Metric] | None) -> str:
    """Create bootstrap header with dynamic metrics configuration.

    Args:
        metrics: Metrics to collect. If None, uses no-op metrics functions.

    Returns:
        Complete header string with metrics collection functions.
    """
    metrics_script = generate_metrics_script(metrics)
    return HEADER_TEMPLATE.format(metrics_script=metrics_script)


def bootstrap(
    *ops: Op | None,
    header: str | None = None,
    metrics: tuple[Metric, ...] | list[Metric] | None = None,
) -> str:
    """Compose operations into a complete bootstrap script.

    Args:
        *ops: Operations to compose. Can be strings or callables returning strings.
        header: Optional custom header. If provided, metrics parameter is ignored.
        metrics: Metrics configuration for dynamic collection. Only used when
            header is not provided.

    Returns:
        Complete shell script string.

    Example:
        >>> from skyward.metrics import CPU, GPU, Default
        >>> script = bootstrap(
        ...     apt("python3", "curl"),
        ...     checkpoint(".step_apt"),
        ...     "echo 'custom command'",
        ...     metrics=Default(),
        ... )
    """
    match (header, metrics):
        case (str(), _):
            base = header
        case (None, list() | tuple() as m):
            base = make_header(m)
        case _:
            from ..metrics import Default

            base = make_header(Default())

    commands = [resolve(op) for op in ops]
    return base + "\n\n".join(commands)
