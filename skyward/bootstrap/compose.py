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

export DEBIAN_FRONTEND=noninteractive
export PATH="/root/.local/bin:$PATH"

# =============================================================================
# Events JSONL Streaming (bootstrap + metrics)
# =============================================================================

EVENTS_LOG="{SKYWARD_DIR}/events.jsonl"
MAX_SIZE=10485760  # 10MB rotation threshold

# Base emit with automatic rotation
emit() {{
    echo "$1" >> "$EVENTS_LOG"
    local size=$(stat -c%s "$EVENTS_LOG" 2>/dev/null || stat -f%z "$EVENTS_LOG" 2>/dev/null || echo 0)
    if [ "$size" -gt "$MAX_SIZE" ]; then
        mv "$EVENTS_LOG" "${{EVENTS_LOG}}.1"
    fi
}}

emit_phase() {{
    local event="$1" phase="$2" elapsed="${{3:-null}}" error="${{4:-null}}"
    emit "{{\\"type\\":\\"phase\\",\\"event\\":\\"$event\\",\\"phase\\":\\"$phase\\",\\"elapsed\\":$elapsed,\\"error\\":$error}}"
}}

emit_console() {{
    local content="$1" stream="${{2:-stdout}}"
    # Escape for JSON: backslash first, then quotes, then control chars
    content="${{content//\\\\/\\\\\\\\}}"
    content="${{content//\\"/\\\\\\"}}"
    content="${{content//$'\\n'/\\\\n}}"
    content="${{content//$'\\t'/\\\\t}}"
    content="${{content//$'\\r'/\\\\r}}"
    emit "{{\\"type\\":\\"console\\",\\"content\\":\\"$content\\",\\"stream\\":\\"$stream\\"}}"
}}

emit_command() {{
    local cmd="$1"
    # Escape for JSON: backslash first, then quotes, then control chars
    cmd="${{cmd//\\\\/\\\\\\\\}}"
    cmd="${{cmd//\\"/\\\\\\"}}"
    cmd="${{cmd//$'\\n'/\\\\n}}"
    cmd="${{cmd//$'\\t'/\\\\t}}"
    cmd="${{cmd//$'\\r'/\\\\r}}"
    emit "{{\\"type\\":\\"command\\",\\"command\\":\\"$cmd\\"}}"
}}

run_phase() {{
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
    local exit_code=${{PIPESTATUS[0]}}
    set -e

    local end_ns=$(date +%s%N)
    local elapsed=$(awk "BEGIN {{printf \\"%.2f\\", ($end_ns - $start_ns) / 1000000000}}")

    if [ $exit_code -eq 0 ]; then
        emit_phase "completed" "$phase" "$elapsed"
    else
        emit_phase "failed" "$phase" "$elapsed" "\\"exit_code=$exit_code\\""
        exit $exit_code
    fi
}}

# =============================================================================
# Metrics Streaming
# =============================================================================

_GPU_CACHE=""
_GPU_COUNT=0

emit_metrics() {{
    local ts=$(date +%s.%N)
    local cpu=$(awk '/^cpu / {{printf "%.1f", ($2+$4)*100/($2+$4+$5)}}' /proc/stat)
    local mem_data=$(free | awk '/^Mem:/ {{printf "%.1f,%d,%d", $3/$2*100, $3/1024, $2/1024}}')
    IFS=',' read -r mem mem_used mem_total <<< "$mem_data"

    # GPU only every 5 iterations (~1s) because nvidia-smi is slow
    if [ $((_GPU_COUNT % 5)) -eq 0 ] && command -v nvidia-smi &>/dev/null; then
        local gpu_raw=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
                        --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
        if [ -n "$gpu_raw" ]; then
            IFS=',' read -r gu gmu gmt gt <<< "$gpu_raw"
            _GPU_CACHE=",\\"gpu_util\\":$gu,\\"gpu_mem_used\\":$gmu,\\"gpu_mem_total\\":$gmt,\\"gpu_temp\\":$gt"
        fi
    fi
    _GPU_COUNT=$((_GPU_COUNT + 1))

    emit "{{\\"type\\":\\"metrics\\",\\"ts\\":$ts,\\"cpu\\":$cpu,\\"mem\\":$mem,\\"mem_used_mb\\":$mem_used,\\"mem_total_mb\\":$mem_total$_GPU_CACHE}}"
}}

start_metrics_daemon() {{
    (
        while true; do
            emit_metrics
            sleep 0.2
        done
    ) &
    echo $! > {SKYWARD_DIR}/metrics.pid
}}

stop_metrics_daemon() {{
    if [ -f {SKYWARD_DIR}/metrics.pid ]; then
        kill $(cat {SKYWARD_DIR}/metrics.pid) 2>/dev/null || true
        rm -f {SKYWARD_DIR}/metrics.pid
    fi
}}

# Error trap - emit failure event
trap 'emit_phase "failed" "bootstrap" null "\\"$BASH_COMMAND failed\\""' ERR

# Start bootstrap
emit_phase "started" "bootstrap"

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
