"""Drives every route against the mock services and dumps the generated OpenAPI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from litestar.testing import TestClient

from skyward.server.app import create_app

HASH = "a" * 64
IDEM = {"Idempotency-Key": "k-1"}
MATCH = {"If-Match": '"7"'}
BLOB = "application/vnd.skyward.blob"


def main() -> int:
    app = create_app()
    failures: list[str] = []

    with TestClient(app=app) as client:
        calls = [
            ("GET", "/v1/computes", {}),
            ("POST", "/v1/computes", {"headers": IDEM, "json": {"spec": {"specs": [{"provider": {"kind": "aws"}}], "nodes": {"desired": 4}}}}),
            ("GET", "/v1/computes/cmp_7f3a1c", {}),
            ("PATCH", "/v1/computes/cmp_7f3a1c", {"headers": MATCH, "json": {"nodes": {"desired": 8}}}),
            ("DELETE", "/v1/computes/cmp_7f3a1c", {"headers": {**IDEM, **MATCH}}),
            ("GET", "/v1/computes/cmp_7f3a1c/generations", {}),
            ("GET", "/v1/computes/cmp_7f3a1c/generations/3", {}),
            ("POST", "/v1/computes/cmp_7f3a1c/generations", {"headers": {**IDEM, **MATCH}, "json": {"force": False}}),
            ("PUT", "/v1/computes/cmp_7f3a1c/lease", {"json": {"owner": "ctl_1", "ttl_seconds": 30}}),
            ("DELETE", "/v1/computes/cmp_7f3a1c/lease", {}),
            ("GET", "/v1/computes/cmp_7f3a1c/nodes", {}),
            ("GET", "/v1/computes/cmp_7f3a1c/nodes/nod_c19e40", {}),
            ("DELETE", "/v1/computes/cmp_7f3a1c/nodes/nod_c19e40", {"headers": IDEM}),
            ("GET", "/v1/functions", {}),
            ("HEAD", f"/v1/functions/{HASH}", {}),
            ("PUT", f"/v1/functions/{HASH}", {"content": b"\x00pickle", "headers": {"Content-Type": BLOB}}),
            ("GET", f"/v1/functions/{HASH}", {}),
            ("HEAD", f"/v1/blobs/{HASH}", {}),
            ("PUT", f"/v1/blobs/{HASH}", {"content": b"\x00args", "headers": {"Content-Type": BLOB}}),
            ("GET", f"/v1/blobs/{HASH}", {}),
            ("GET", "/v1/tasks", {}),
            ("POST", "/v1/tasks", {"headers": IDEM, "json": {"compute": "training", "function": HASH, "dispatch": "one"}}),
            ("GET", "/v1/tasks/tsk_9d21f0", {}),
            ("DELETE", "/v1/tasks/tsk_9d21f0", {"headers": IDEM}),
            ("GET", "/v1/tasks/tsk_9d21f0/result", {}),
            ("GET", "/v1/tasks/tsk_9d21f0/executions", {}),
            ("GET", "/v1/tasks/tsk_9d21f0/executions/1", {}),
            ("POST", "/v1/tasks/tsk_9d21f0/executions", {"headers": IDEM, "json": {"acknowledge_duplication": True}}),
            ("GET", "/v1/providers", {}),
            ("GET", "/v1/offers", {}),
            ("GET", "/v1/health/live", {}),
            ("GET", "/v1/health/ready", {}),
            ("GET", "/v1/health/dependencies", {}),
        ]

        for method, path, kwargs in calls:
            response = client.request(method, path, **kwargs)
            ok = response.status_code < 300
            print(f"{'ok ' if ok else 'FAIL'} {method:6} {path:52} {response.status_code}")
            if not ok:
                failures.append(f"{method} {path} -> {response.status_code} {response.text[:200]}")

        schema = client.get("/v1/schema/openapi.json")
        Path("openapi.json").write_text(json.dumps(schema.json(), indent=2, sort_keys=True))
        paths = schema.json()["paths"]
        print(f"\nopenapi: {len(paths)} paths, {sum(len(v) for v in paths.values())} operations -> v2/openapi.json")

    for failure in failures:
        print(f"\n{failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
