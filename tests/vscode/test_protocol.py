import json

import pytest

from vscode.sidecar.protocol import Request, format_event, format_response, parse_request


def test_parse_request_valid():
    line = '{"id": 1, "method": "pools/list", "params": {}}'
    req = parse_request(line)
    assert req.id == 1
    assert req.method == "pools/list"
    assert req.params == {}


def test_parse_request_missing_method():
    with pytest.raises(ValueError, match="method"):
        parse_request('{"id": 1}')


def test_parse_request_defaults():
    line = '{"method": "pools/list"}'
    req = parse_request(line)
    assert req.id == 0
    assert req.method == "pools/list"
    assert req.params == {}


def test_parse_request_invalid_json():
    with pytest.raises(json.JSONDecodeError):
        parse_request("not json")


def test_parse_request_is_frozen():
    req = parse_request('{"id": 1, "method": "pools/list", "params": {}}')
    with pytest.raises(AttributeError):
        req.id = 2  # type: ignore[misc]


def test_format_response():
    out = format_response(1, {"pools": []})
    parsed = json.loads(out)
    assert parsed == {"id": 1, "result": {"pools": []}}


def test_format_error():
    out = format_response(1, error="not found")
    parsed = json.loads(out)
    assert parsed == {"id": 1, "error": "not found"}


def test_format_response_null_result():
    out = format_response(1)
    parsed = json.loads(out)
    assert parsed == {"id": 1, "result": None}


def test_format_event():
    out = format_event("task.completed", "train", {"task_id": "t1"})
    parsed = json.loads(out)
    assert parsed == {"event": "task.completed", "pool": "train", "data": {"task_id": "t1"}}


def test_request_dataclass():
    req = Request(id=5, method="pools/status", params={"name": "train"})
    assert req.id == 5
    assert req.method == "pools/status"
    assert req.params == {"name": "train"}
