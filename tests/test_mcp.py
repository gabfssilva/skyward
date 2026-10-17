"""``sky mcp``, driven the way an assistant's host drives it.

The server runs as its own process, spoken to over stdin and stdout by the
protocol's own client, against a daemon of its own. Nothing is patched: what is
asserted is what the assistant would read, and what the daemon then holds.
"""

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

pytest.importorskip("mcp", reason="sky mcp needs: pip install 'skyward[mcp]'")

from mcp import Client, StdioServerParameters  # noqa: E402
from mcp.types import TextContent  # noqa: E402

import skyward as sky  # noqa: E402
from skyward.cli.mcp import HIDDEN  # noqa: E402
from skyward.server.persistence.metrics import MetricStore  # noqa: E402
from skyward.shared.schemas import MetricSample  # noqa: E402
from tests.conftest import SKY, given, serving  # noqa: E402
from tests.test_script import declared, entries  # noqa: E402

READS = {
    "list_computes",
    "get_compute",
    "list_tasks",
    "read_log",
    "read_metrics",
    "read_metric_history",
    "read_daemon_log",
    "list_offers",
    "list_providers",
    "read_result",
}
MINUTE = 60_000
LATELY = time.time_ns() // 1_000_000 // MINUTE * MINUTE - 3 * MINUTE
"""The start of a minute three minutes ago: the recent samples share its step."""
LONG_AGO = LATELY - 180 * MINUTE
CHANGES = {"scale_compute", "update_compute", "delete_compute", "cancel_task", "exec_command"}
BUYS = {"create_compute", "run_script"}
TRAIN = "import skyward as sky\n\n\n@sky.app(provider=sky.AWS())\ndef train(epochs: int = 10) -> float:\n    return 0.5\n"
"""A file whose one function runs on AWS: read, it is parsed and refused here, and nothing is bought."""


@pytest.mark.local
def describe_the_tools() -> None:
    async def every_operation_is_offered_with_what_it_does(alone: str) -> None:
        async with connected(alone) as client:
            tools = (await client.list_tools()).tools

        assert {tool.name for tool in tools} == READS | CHANGES | BUYS
        assert all(tool.description for tool in tools), "the description is all an assistant knows about a tool"

    async def the_reads_say_they_only_read_and_the_rest_what_they_do_to_machines(alone: str) -> None:
        async with connected(alone) as client:
            hints = {tool.name: tool.annotations for tool in (await client.list_tools()).tools}

        assert {name for name, hint in hints.items() if hint and hint.read_only_hint} == READS
        assert {name for name, hint in hints.items() if hint and hint.destructive_hint} == CHANGES
        assert all((hint := hints[name]) and hint.read_only_hint is False for name in BUYS), "buying machines is not a read"


@pytest.mark.local
def describe_reading() -> None:
    def describe_when_the_daemon_has_no_computes() -> None:
        async def listing_them_is_an_empty_answer_not_a_failure(alone: str) -> None:
            async with connected(alone) as client:
                assert await answer(client, "list_computes") == []

    def describe_when_the_compute_named_is_not_there() -> None:
        @pytest.mark.parametrize(("tool", "arguments"), [("get_compute", {"ref": "absent"}), ("cancel_task", {"task": "absent"})])
        async def the_refusal_is_the_daemons_by_name(alone: str, tool: str, arguments: dict[str, str]) -> None:
            async with connected(alone) as client:
                said = await refusal(client, tool, **arguments)

            assert "not_found" in said

    def describe_offers() -> None:
        async def the_catalog_is_cut_to_the_limit_and_says_how_much_matched(alone: str) -> None:
            async with connected(alone) as client:
                await answer(client, "create_compute", provider="fake")
                cut = await answer(client, "list_offers", limit=2)
                h100 = await answer(client, "list_offers", accelerator="h100")

            assert len(cut["items"]) == 2
            assert cut["total"] > 2, "the rest of the catalog is counted, not sent"
            assert h100["items"], "the fake catalog has h100s"
            assert {offer["accelerator"] for offer in h100["items"]} == {"h100"}

    def describe_provider_kinds() -> None:
        async def they_name_what_each_kind_reads_from_the_environment(alone: str) -> None:
            async with connected(alone) as client:
                kinds = {kind["kind"]: kind.get("credential_fields", []) for kind in await answer(client, "list_providers", kinds=True)}

            assert kinds["fake"] == [], "a kind that reads nothing has no credential_fields at all"
            assert kinds["runpod"] == ["api_key"]

    def describe_the_event_log() -> None:
        async def it_is_narrowed_to_every_type_asked_for(alone: str) -> None:
            async with connected(alone) as client:
                await answer(client, "create_compute", provider="fake", name="logged")
                entries = (await answer(client, "read_log", compute="logged", types=["compute.created"]))["items"]

            assert entries, "a compute that was created has said so"
            assert {entry["type"] for entry in entries} == {"compute.created"}
            assert all("compute" not in entry for entry in entries), "the filter named the compute; every entry would repeat it"


@pytest.mark.local
def describe_an_answer() -> None:
    async def it_holds_nothing_null_or_empty(alone: str) -> None:
        async with connected(alone) as client:
            await answer(client, "create_compute", provider="fake", name="brief")
            answers = {
                "get_compute": await answer(client, "get_compute", ref="brief"),
                "list_offers": await answer(client, "list_offers", limit=5),
                "read_log": await answer(client, "read_log", compute="brief"),
                "list_providers": await answer(client, "list_providers"),
            }

        assert {tool: emptied(said) for tool, said in answers.items()} == dict.fromkeys(answers, [])

    async def its_times_are_to_the_second(alone: str) -> None:
        async with connected(alone) as client:
            await answer(client, "create_compute", provider="fake", name="timed")
            answers = [
                await answer(client, "get_compute", ref="timed"),
                await answer(client, "read_log", compute="timed"),
                await answer(client, "read_daemon_log", compute="timed"),
            ]

        spelled = [time for said in answers for time in times(said)]
        assert spelled, "every one of these answers says when something happened"
        assert [time for time in spelled if not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", time)] == []

    async def raw_is_the_daemons_resource_whole(alone: str) -> None:
        async with connected(alone) as client:
            await answer(client, "create_compute", provider="fake", name="whole")
            brief = await answer(client, "get_compute", ref="whole")
            raw = await answer(client, "get_compute", ref="whole", raw=True)

        bookkeeping = {"spec", "revision", "generation", "lease"}
        assert bookkeeping <= raw.keys()
        assert not bookkeeping & brief.keys()
        assert raw["spec"]["nodes"] == {"initial": 1, "min": None, "max": None}
        assert brief["size"] == {"initial": 1}, "the same size, without what is unset"

    def describe_secrets() -> None:
        async def an_environment_value_or_a_plugin_credential_is_never_answered(alone: str) -> None:
            async with connected(alone) as client:
                answers = [
                    await answer(
                        client,
                        "create_compute",
                        provider="fake",
                        name="keyed",
                        env={"HF_HOME": "/data/hf", "API_KEY": "s3cret"},
                        plugins={"huggingface": {"token": "hf_s3cret"}, "torch": {"backend": "gloo"}},
                    ),
                    await answer(client, "get_compute", ref="keyed", raw=True),
                    *await answer(client, "list_computes"),
                    *await answer(client, "list_computes", raw=True),
                ]
                brief = await answer(client, "get_compute", ref="keyed")
            held = httpx.get(f"{alone}/v1/computes/keyed").json()["spec"]

            assert not [said for said in (*answers, brief) if "s3cret" in json.dumps(said)]
            assert brief["image"]["env"] == {"HF_HOME": HIDDEN, "API_KEY": HIDDEN}, "which variables are set is still said"
            plugins = {plugin["kind"]: plugin["params"] for plugin in brief["plugins"]}
            assert plugins["huggingface"]["token"] == HIDDEN
            assert plugins["torch"]["backend"] == "gloo", "a parameter that is not a credential is shown"
            assert held["image"]["env"]["API_KEY"] == "s3cret", "the daemon keeps what the nodes need; only the answer hides it"


@pytest.mark.local
def describe_a_measured_compute() -> None:
    @pytest.fixture
    async def measured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[tuple[str, str]]:
        """A daemon over a compute whose node took two cpu samples a minute ago and one three hours ago."""
        monkeypatch.delenv("SKYWARD_URL", raising=False)
        database = tmp_path / "skyward.sqlite"
        _, compute = await given(database)
        store = MetricStore()
        store.add(compute.id, (sample(LATELY + 1_000, 10.0), sample(LATELY + 2_000, 20.0), sample(LONG_AGO + 1_000, 70.0)))
        await store.flush()
        with serving(database, tmp_path / "daemon.log") as url:
            yield url, compute.id

    async def its_newest_values_come_as_one_object_per_node(measured: tuple[str, str]) -> None:
        url, compute = measured
        async with connected(url) as client:
            latest = await answer(client, "read_metrics", compute=compute)

        assert latest == [{"node": "nod_a", "at": second(LATELY + 2_000), "values": {"cpu": 20.0}}]

    async def the_window_ends_now_and_each_step_is_one_value(measured: tuple[str, str]) -> None:
        url, compute = measured
        async with connected(url) as client:
            averaged = await answer(client, "read_metric_history", compute=compute, name=["cpu"])
            highest = await answer(client, "read_metric_history", compute=compute, name=["cpu"], agg="max")

        assert points(averaged) == [(LATELY, 15.0)], "the sample from three hours ago is outside the last hour"
        assert points(highest) == [(LATELY, 20.0)]

    async def a_window_that_ends_earlier_reads_what_was_measured_then(measured: tuple[str, str]) -> None:
        url, compute = measured
        until = datetime.fromtimestamp((LONG_AGO + 30 * MINUTE) / 1000, tz=UTC).isoformat()
        async with connected(url) as client:
            then = await answer(client, "read_metric_history", compute=compute, name=["cpu"], until=until)

        assert points(then) == [(LONG_AGO, 70.0)]

    async def a_window_cut_finer_than_the_daemon_answers_says_why(measured: tuple[str, str]) -> None:
        url, compute = measured
        async with connected(url) as client:
            said = await refusal(client, "read_metric_history", compute=compute, name=["cpu"], minutes=7 * 24 * 60, step_seconds=1)

        assert "at most 2000 steps" in said


@pytest.mark.local
def describe_the_daemons_own_log() -> None:
    async def it_explains_a_compute_that_cannot_be_provisioned(alone: str) -> None:
        async with connected(alone) as client:
            created = await answer(client, "create_compute", provider="fake", name="unplaced")
            lines: list[dict[str, Any]] = []
            deadline = time.monotonic() + 30
            while not lines and time.monotonic() < deadline:
                await asyncio.sleep(0.5)
                lines = (await answer(client, "read_daemon_log", compute="unplaced", level="WARNING", contains=["cannot provision"]))["items"]

            whole = (await answer(client, "read_daemon_log", compute="unplaced", level="WARNING", contains=["cannot provision"], raw=True))["items"]

        assert lines, "the daemon said why the fake provider bought nothing"
        assert {line["level"] for line in lines} <= {"WARNING", "ERROR"}
        assert {line["compute"] for line in whole} == {created["id"]}, "the filter's compute is left out of the lines, not out of the log"
        assert not any("compute" in line for line in lines)


@pytest.mark.local
def describe_creating_a_compute() -> None:
    async def the_definition_it_was_given_is_the_one_the_daemon_keeps(alone: str) -> None:
        async with connected(alone) as client:
            created = await answer(
                client,
                "create_compute",
                provider="fake",
                name="built",
                python="3.12",
                pip=["torch", "tilelang[nvcc]"],
                apt=["build-essential"],
                pip_indexes=["https://download.pytorch.org/whl/cu128"],
                env={"HF_HOME": "/data/hf"},
                plugins={"torch": {"backend": "gloo"}, "mps": {"active_thread_percentage": 50}, "huggingface": {}},
            )
        spec = httpx.get(f"{alone}/v1/computes/{created['id']}").json()["spec"]

        assert created["name"] == "built"
        assert spec["image"]["python"] == "3.12"
        assert spec["image"]["pip"] == ["torch", "tilelang[nvcc]"]
        assert spec["image"]["apt"] == ["build-essential"]
        assert spec["image"]["pip_indexes"] == [{"url": "https://download.pytorch.org/whl/cu128", "packages": []}]
        assert spec["image"]["env"] == {"HF_HOME": "/data/hf"}
        plugins = {ref["kind"]: ref["params"] for ref in spec["plugins"]}
        assert plugins["torch"]["backend"] == "gloo"
        assert plugins["torch"]["cuda"] == "cu128", "a parameter not given is the plugin's own default, as in the SDK"
        assert plugins["mps"]["active_thread_percentage"] == 50, "a parameter keeps the JSON type it was given"
        assert "huggingface" in plugins

    def describe_when_an_argument_names_something_that_does_not_exist() -> None:
        @pytest.mark.parametrize(
            ("arguments", "said"),
            [
                ({"provider": "nowhere"}, "unknown provider 'nowhere'"),
                ({"provider": "fake", "plugins": {"nowhere": {}}}, "unknown plugin 'nowhere'"),
                ({"provider": "fake", "plugins": {"torch": {"backend": "nowhere"}}}, "plugin torch"),
            ],
        )
        async def it_is_refused_before_anything_is_bought(alone: str, arguments: dict[str, Any], said: str) -> None:
            async with connected(alone) as client:
                assert said in await refusal(client, "create_compute", **arguments)
                assert await answer(client, "list_computes") == [], "nothing was created on the way to the refusal"


@pytest.mark.local
def describe_changing_a_compute() -> None:
    async def scaling_it_to_a_range_makes_it_elastic(alone: str) -> None:
        async with connected(alone) as client:
            await answer(client, "create_compute", provider="fake", name="sized")
            scaled = await answer(client, "scale_compute", ref="sized", nodes=2, max_nodes=4)

        assert scaled["size"] == {"initial": 2, "min": 2, "max": 4}

    @pytest.mark.parametrize(("arguments", "said"), [({"nodes": 4, "max_nodes": 2}, "max_nodes"), ({"nodes": -1}, "greater than or equal to 0")])
    async def a_size_that_cannot_be_is_refused_before_it_is_written(alone: str, arguments: dict[str, int], said: str) -> None:
        async with connected(alone) as client:
            await answer(client, "create_compute", provider="fake", name="sized")
            refused = await refusal(client, "scale_compute", ref="sized", **arguments)
            kept = await answer(client, "get_compute", ref="sized")

        assert said in refused
        assert kept["size"]["initial"] == 1, "the size it had is the size it keeps"

    async def a_fixed_image_refuses_new_packages(alone: str) -> None:
        async with connected(alone) as client:
            await answer(client, "create_compute", provider="fake", name="fixed", pip=["torch"])
            said = await refusal(client, "update_compute", ref="fixed", pip=["jax"])

        assert "image_fixed" in said

    async def a_mutable_image_takes_them(alone: str) -> None:
        async with connected(alone) as client:
            await answer(client, "create_compute", provider="fake", name="mutable", pip=["torch"], mutable=True)
            updated = await answer(client, "update_compute", ref="mutable", pip=["jax"])

        assert updated["image"]["pip"] == ["jax"]

    async def deleting_it_is_accepted_not_done(alone: str) -> None:
        async with connected(alone) as client:
            await answer(client, "create_compute", provider="fake", name="doomed")
            deleted = await answer(client, "delete_compute", ref="doomed")

        assert deleted["state"] in ("deleting", "deleted")


@pytest.mark.local
def describe_running_a_script() -> None:
    @pytest.mark.parametrize(
        ("source", "args", "said"),
        [
            (None, [], "no such file"),
            ("print('hello')\n", [], "declares no compute"),
            (TRAIN, ["--epochs", "five"], "--epochs"),
        ],
    )
    async def what_cannot_run_is_refused_before_anything_is_bought(alone: str, tmp_path: Path, source: str | None, args: list[str], said: str) -> None:
        path = tmp_path / "train.py"
        if source is not None:
            path.write_text(source)
        async with connected(alone) as client:
            refused = await refusal(client, "run_script", path=str(path), args=args)
            listed = await answer(client, "list_computes")

        assert said in refused
        assert listed == []

    async def help_answers_with_the_functions_and_what_they_take(alone: str, tmp_path: Path) -> None:
        path = tmp_path / "train.py"
        path.write_text(TRAIN)
        async with connected(alone) as client:
            helped = await text(client, "run_script", path=str(path), args=["--help"])
            listed = await answer(client, "list_computes")

        assert "--epochs" in helped, "what the file's function takes, as sky run would print it"
        assert listed == []


@pytest.mark.local
def describe_when_no_daemon_is_running() -> None:
    async def a_tool_says_where_it_looked_and_how_to_start_one() -> None:
        async with connected("http://127.0.0.1:1") as client:
            said = await refusal(client, "list_computes")

        assert "no daemon at http://127.0.0.1:1" in said
        assert "sky server start" in said


@pytest.mark.compute
@pytest.mark.xdist_group("pool")
def describe_running_a_command_on_the_nodes() -> None:
    async def every_node_answers_with_what_it_printed(daemon: str, pool: sky.Compute) -> None:
        async with connected(daemon) as client:
            ran = await answer(client, "exec_command", ref="shared", command="echo alive")

        assert len(ran) == pool.current_nodes()
        assert {(result["exit_code"], result["stdout"].strip()) for result in ran.values()} == {(0, "alive")}

    async def a_task_comes_with_its_attempts_and_without_hashes(daemon: str, pool: sky.Compute) -> None:
        assert noted() >> pool == "noted"
        async with connected(daemon) as client:
            (task,) = (await answer(client, "list_tasks", compute="shared", state="succeeded", limit=1))["items"]

        assert task["function"]["name"] == "noted"
        assert [attempt["state"] for attempt in task["executions"]] == ["succeeded"]
        assert not {"args_sha256", "result_sha256", "retry"} & task.keys()


@pytest.mark.compute
@pytest.mark.xdist_group("script")
def describe_a_script_run_on_containers() -> None:
    async def a_file_runs_on_the_compute_it_declares_which_goes_when_it_ends(daemon: str, tmp_path: Path) -> None:
        path = tmp_path / "declared.py"
        path.write_text(declared(nodes=1, delete_on_exit=True))
        async with connected(daemon) as client:
            started = await answer(client, "run_script", path=str(path), args=["3"])
            task = await until(lambda: ended(client, started["name"]))
            result = await answer(client, "read_result", task=task["id"])
            printed = await answer(client, "read_log", task=task["id"], types=["node.console"])
            gone = await until(lambda: state(client, started["name"], "deleted"))

        assert started["delete_on_exit"] is True
        assert Path(started["log"]).is_file(), "sky run's own progress is kept where the answer says"
        assert result == {"state": "succeeded", "results": [{"rank": 0, "returned": 3}]}, "a file run whole returns its exit status"
        assert "rank 0 of 1 said ['3']" in [entry.get("content") for entry in printed["items"]]
        assert gone == "deleted", "the run took the compute down once its task ended, as the file declared"

    async def a_function_answers_with_what_it_returned(daemon: str, tmp_path: Path) -> None:
        path = tmp_path / "shouting.py"
        path.write_text(entries(nodes=1, delete_on_exit=False))
        async with connected(daemon) as client:
            started = await answer(client, "run_script", path=str(path), args=["shout", "--word", "hi", "--times", "2"])
            try:
                task = await until(lambda: ended(client, started["name"]))
                result = await answer(client, "read_result", task=task["id"])
                kept = await answer(client, "get_compute", ref=started["name"])
            finally:
                await answer(client, "delete_compute", ref=started["name"])

        assert result == {"state": "succeeded", "results": [{"rank": 0, "returned": {"rank": 0, "said": "hihi"}}]}
        assert kept["state"] == "ready", "delete_on_exit = false leaves it up for the next run"


@sky.function
def noted() -> str:
    return "noted"


async def until[T](probe: Callable[[], Awaitable[T | None]], seconds: float = 300) -> T:
    """What ``probe`` finds, once it finds something."""
    deadline = time.monotonic() + seconds
    while (found := await probe()) is None:
        assert time.monotonic() < deadline, f"nothing turned up in {seconds:.0f}s"
        await asyncio.sleep(1)
    return found


async def ended(client: Client, compute: str) -> dict[str, Any] | None:
    """The compute's task, once it is no longer queued or running."""
    tasks = (await answer(client, "list_tasks", compute=compute))["items"]
    return next((task for task in tasks if task["state"] not in ("queued", "running")), None)


async def state(client: Client, compute: str, *wanted: str) -> str | None:
    """The compute's state, once it is one of ``wanted``."""
    current = (await answer(client, "get_compute", ref=compute))["state"]
    return current if current in wanted else None


def sample(at: int, value: float) -> MetricSample:
    return MetricSample(node="nod_a", name="cpu", at=at, value=value)


def second(ms: int) -> str:
    """A time as an answer spells it: UTC, to the second."""
    return datetime.fromtimestamp(ms / 1000, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def times(value: object) -> list[str]:
    """Every string in ``value`` that spells a moment."""
    match value:
        case dict():
            return [time for item in value.values() for time in times(item)]
        case list():
            return [time for item in value for time in times(item)]
        case str() if re.match(r"\d{4}-\d\d-\d\dT\d\d:\d\d", value):
            return [value]
        case _:
            return []


def emptied(value: object, at: str = "") -> list[str]:
    """Where in ``value`` something is null or empty."""
    match value:
        case dict():
            return [path for key, item in value.items() for path in ([f"{at}.{key}"] if item in (None, "", [], {}) else emptied(item, f"{at}.{key}"))]
        case list():
            return [path for index, item in enumerate(value) for path in emptied(item, f"{at}[{index}]")]
        case _:
            return []


def points(history: dict[str, Any]) -> list[tuple[int, float]]:
    return [(at, value) for series in history["series"] for at, value in zip(series["at"], series["values"], strict=True)]


@asynccontextmanager
async def connected(url: str) -> AsyncIterator[Client]:
    async with Client(StdioServerParameters(command=str(SKY), args=["mcp", "--url", url])) as client:
        yield client


async def answer(client: Client, tool: str, **arguments: object) -> Any:
    """What the tool answered, read back from its JSON."""
    result = await client.call_tool(tool, arguments)
    match result.content:
        case [TextContent(text=text)] if not result.is_error:
            return json.loads(text)
        case _:
            raise AssertionError(f"{tool} was refused: {result.content}")


async def text(client: Client, tool: str, **arguments: object) -> str:
    """What the tool answered, as it said it."""
    result = await client.call_tool(tool, arguments)
    match result.content:
        case [TextContent(text=said)] if not result.is_error:
            return said
        case _:
            raise AssertionError(f"{tool} was refused: {result.content}")


async def refusal(client: Client, tool: str, **arguments: object) -> str:
    """What the tool said when it refused."""
    result = await client.call_tool(tool, arguments)
    match result.content:
        case [TextContent(text=text)] if result.is_error:
            return text
        case _:
            raise AssertionError(f"{tool} answered where it should have refused: {result.content}")
