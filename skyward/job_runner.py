"""Ray Job runner script for Skyward.

This script is the entrypoint for Ray Jobs submitted by the Executor.
It handles:
1. Reading serialized payload (function + args) from env var or file
2. Deserializing using cloudpickle
3. Executing with optional node placement via Ray resources
4. Serializing result to file for retrieval
"""

from __future__ import annotations

import base64
import io
import os
import sys
from contextlib import contextmanager
from typing import Any, TextIO

# Path to events log file (same as bootstrap uses)
EVENTS_LOG = "/opt/skyward/events.jsonl"


def _emit_log(content: str, stream: str = "stdout") -> None:
    """Emit a log event to events.jsonl."""
    # Escape for JSON
    escaped = content.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    line = f'{{"type":"log","content":"{escaped}","stream":"{stream}"}}\n'
    try:
        with open(EVENTS_LOG, "a") as f:
            f.write(line)
    except Exception:
        pass  # Don't fail if we can't write to log


class _TeeWriter(io.TextIOBase):
    """Writer that tees output to both original stream and events.jsonl."""

    def __init__(self, original: TextIO, stream_name: str = "stdout"):
        self._original = original
        self._stream_name = stream_name

    def write(self, s: str) -> int:
        # Write to original
        result = self._original.write(s)
        self._original.flush()
        # Also emit to events.jsonl (strip trailing newlines for cleaner logs)
        if s and s.strip():
            _emit_log(s.rstrip("\n"), self._stream_name)
        return result

    def flush(self) -> None:
        self._original.flush()


@contextmanager
def _capture_output():
    """Context manager that tees stdout/stderr to events.jsonl."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = _TeeWriter(old_stdout, "stdout")  # type: ignore[assignment]
        sys.stderr = _TeeWriter(old_stderr, "stderr")  # type: ignore[assignment]
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def _deserialize_payload() -> dict[str, Any]:
    """Read and deserialize the job payload.

    Payload can come from:
    - SKYWARD_PAYLOAD env var (base64 encoded, for small payloads)
    - File path as first argument (for large payloads)
    """
    from skyward.utils.serialization import deserialize

    # Try env var first (small payloads)
    payload_b64 = os.environ.get("SKYWARD_PAYLOAD")
    if payload_b64:
        payload = base64.b64decode(payload_b64)
        return deserialize(payload)

    # Fall back to file (large payloads)
    if len(sys.argv) > 1:
        payload_file = sys.argv[1]
        with open(payload_file, "rb") as f:
            return deserialize(f.read())

    raise RuntimeError("No payload found in SKYWARD_PAYLOAD env var or file argument")


def _get_node_gpu_count(node_id: int) -> int:
    """Get the number of GPUs available on a specific node."""
    import ray

    # Get all nodes and find the one with our resource
    for node in ray.nodes():
        if node.get("Alive") and f"node_{node_id}" in node.get("Resources", {}):
            return int(node.get("Resources", {}).get("GPU", 0))
    return 0


def _execute_single(
    fn: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    node_id: int | None,
    num_nodes: int = 1,
    pool_infos: list[str] | None = None,
) -> Any:
    """Execute function on a single node with all available GPUs."""
    if node_id is not None:
        import ray

        if not ray.is_initialized():
            ray.init()

        num_gpus = _get_node_gpu_count(node_id)
        head_addr = os.environ.get("SKYWARD_HEAD_ADDR", "127.0.0.1")
        pool_json = pool_infos[node_id] if pool_infos and node_id < len(pool_infos) else ""

        @ray.remote(resources={f"node_{node_id}": 1}, num_gpus=num_gpus, max_retries=0)
        def run_fn(nid: int, total: int, head: str, pool: str) -> Any:
            if pool:
                os.environ["COMPUTE_POOL"] = pool
            os.environ["SKYWARD_NODE_ID"] = str(nid)
            os.environ["SKYWARD_TOTAL_NODES"] = str(total)
            os.environ["SKYWARD_HEAD_ADDR"] = head
            os.environ["SKYWARD_HEAD_PORT"] = "29500"
            with _capture_output():
                return fn(*args, **kwargs)

        return ray.get(run_fn.remote(node_id, num_nodes, head_addr, pool_json))
    else:
        return fn(*args, **kwargs)


def _execute_broadcast(
    fn: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    num_nodes: int,
    pool_infos: list[str] | None = None,
) -> list[Any]:
    """Execute function on all nodes with all available GPUs."""
    import ray

    if not ray.is_initialized():
        ray.init()

    head_addr = os.environ.get("SKYWARD_HEAD_ADDR", "127.0.0.1")

    refs = []
    for nid in range(num_nodes):
        num_gpus = _get_node_gpu_count(nid)
        pool_json = pool_infos[nid] if pool_infos and nid < len(pool_infos) else ""

        @ray.remote(resources={f"node_{nid}": 1}, num_gpus=num_gpus, max_retries=0)
        def run_on_node(node_id: int, total: int, head: str, pool: str) -> Any:
            if pool:
                os.environ["COMPUTE_POOL"] = pool
            os.environ["SKYWARD_NODE_ID"] = str(node_id)
            os.environ["SKYWARD_TOTAL_NODES"] = str(total)
            os.environ["SKYWARD_HEAD_ADDR"] = head
            os.environ["SKYWARD_HEAD_PORT"] = "29500"
            with _capture_output():
                return fn(*args, **kwargs)

        refs.append(run_on_node.remote(nid, num_nodes, head_addr, pool_json))

    return ray.get(refs)


def _write_result(result: Any, result_file: str) -> None:
    """Serialize and write result to file."""
    from skyward.utils.serialization import serialize

    with open(result_file, "wb") as f:
        f.write(serialize(result))


def main() -> None:
    """Main entry point for job execution."""
    # Get result file path from env
    result_file = os.environ.get("SKYWARD_RESULT_FILE", "/tmp/skyward_result.pkl")

    try:
        # Load payload
        data = _deserialize_payload()

        fn = data["fn"]
        args = data.get("args", ())
        kwargs = data.get("kwargs", {})
        mode = data.get("mode", "single")

        # Execute based on mode
        pool_infos = data.get("pool_infos", [])
        if mode == "broadcast":
            num_nodes = data["num_nodes"]
            result = _execute_broadcast(fn, args, kwargs, num_nodes, pool_infos)
        else:
            node_id = data.get("node_id")
            num_nodes = data.get("num_nodes", 1)
            result = _execute_single(fn, args, kwargs, node_id, num_nodes, pool_infos)

        # Write result
        _write_result(result, result_file)

        # Print result file path for logging
        print(f"SKYWARD_RESULT_FILE={result_file}")
        print("SKYWARD_JOB_SUCCESS")

    except Exception as e:
        # Write error as result
        import traceback

        error_info = {
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        _write_result(error_info, result_file)

        print(f"SKYWARD_RESULT_FILE={result_file}")
        print(f"SKYWARD_JOB_ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
