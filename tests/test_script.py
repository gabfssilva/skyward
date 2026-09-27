"""A script that declares its own compute, and the node that runs it.

``sky run`` reads the PEP 723 block at the top of a file: ``requires-python`` and
``dependencies`` are the image, ``[tool.skyward]`` the rest of the compute. What is
asserted here is what the header comes to before any daemon is asked for anything,
what the node does with the text once it has it, and — on containers — what a run
does with the compute: create it, keep it, resize it and take it down.
"""

from __future__ import annotations

import textwrap
from functools import partial
from pathlib import Path

import cloudpickle
import httpx
import pytest

from skyward.cli.script import read
from skyward.shared.schemas import NodeBounds, PipIndex
from skyward.worker.plugins import HuggingFace, Torch
from skyward.worker.script import run
from skyward.worker.stopping import Stop
from tests.conftest import PYTHON, cli

HEADER = """
# /// script
# requires-python = ">=3.12"
# dependencies = ["safetensors", "numpy<3"]
#
# [tool.skyward]
# provider = "aws"
# accelerator = "A100"
# allocation = "spot"
#
# [tool.skyward.nodes]
# initial = 4
# min = 2
#
# [tool.skyward.image]
# apt = ["htop"]
# env = { NCCL_DEBUG = "WARN" }
# pip_indexes = [{ url = "https://download.pytorch.org/whl/cu128", packages = ["torch"] }]
#
# [[tool.skyward.plugins]]
# kind = "torch"
# backend = "gloo"
#
# [[tool.skyward.plugins]]
# kind = "huggingface"
# ///

print("hello")
"""


def written(tmp_path: Path, text: str, name: str = "train.py") -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(text).lstrip())
    return path


def header(tmp_path: Path, skyward: str, top: str = "", name: str = "train.py") -> Path:
    """A script whose block holds ``top`` above a ``[tool.skyward]`` table holding ``skyward``."""
    lines = [*top.strip().splitlines(), "[tool.skyward]", *skyward.strip().splitlines()]
    block = "\n".join(["# /// script", *(f"# {line}" if line else "#" for line in lines), "# ///"])
    return written(tmp_path, f"{block}\nprint('hello')\n", name)


@pytest.mark.local
def describe_reading_a_header() -> None:
    def it_is_the_compute_sky_compute_would_be_given(tmp_path: Path) -> None:
        script = read(written(tmp_path, HEADER))

        assert (script.provider, script.accelerator, script.allocation) == ("aws", "A100", "spot")
        assert script.nodes == NodeBounds(initial=4, min=2)
        assert script.image.python == "3.12"
        assert tuple(script.image.pip) == ("safetensors", "numpy<3")
        assert tuple(script.image.apt) == ("htop",)
        assert script.image.env == {"NCCL_DEBUG": "WARN"}
        assert tuple(script.image.pip_indexes) == (PipIndex(url="https://download.pytorch.org/whl/cu128", packages=("torch",)),)
        assert script.plugins == (Torch(backend="gloo"), HuggingFace())
        assert script.delete_on_exit is True
        assert script.source.endswith('print("hello")\n')

    def describe_requires_python() -> None:
        @pytest.mark.parametrize(
            ("specifier", "chosen"),
            [(">=3.12", "3.12"), (">=3.13", "3.13"), ("==3.14.*", "3.14"), ("~=3.12", "3.12"), (">=3.12.4", "3.13")],
        )
        def it_is_the_first_interpreter_it_admits(tmp_path: Path, specifier: str, chosen: str) -> None:
            assert read(header(tmp_path, 'provider = "aws"', f'requires-python = "{specifier}"')).image.python == chosen

        def left_out_it_is_the_image_default(tmp_path: Path) -> None:
            assert read(header(tmp_path, 'provider = "aws"')).image.python is None

        @pytest.mark.parametrize(("specifier", "said"), [("<3.12", "admits none of"), ("three", "not a version specifier")])
        def it_refuses_one_no_node_can_satisfy(tmp_path: Path, specifier: str, said: str) -> None:
            with pytest.raises(SystemExit, match=said):
                read(header(tmp_path, 'provider = "aws"', f'requires-python = "{specifier}"'))

    def describe_nodes() -> None:
        def a_count_is_a_fixed_size(tmp_path: Path) -> None:
            assert read(header(tmp_path, 'provider = "aws"\nnodes = 3')).nodes == NodeBounds(initial=3)

        def a_table_without_initial_opens_at_its_floor(tmp_path: Path) -> None:
            assert read(header(tmp_path, 'provider = "aws"\nnodes = { min = 2, max = 8 }')).nodes == NodeBounds(initial=2, min=2, max=8)

        @pytest.mark.parametrize(
            ("nodes", "said"),
            [
                ("{ initial = 4, min = 6 }", "min <= initial <= max"),
                ("{ initial = 4, max = 2 }", "min <= initial <= max"),
                ("{ max = 8 }", "an initial or a min"),
                ("0", "at least one"),
            ],
        )
        def it_refuses_bounds_that_disagree_with_themselves(tmp_path: Path, nodes: str, said: str) -> None:
            with pytest.raises(SystemExit, match=said):
                read(header(tmp_path, f'provider = "aws"\nnodes = {nodes}'))

    def describe_plugins() -> None:
        @pytest.mark.parametrize(
            ("plugin", "said"),
            [
                ('{ kind = "nowhere" }', "unknown plugin 'nowhere'"),
                ('{ kind = "torch", backend = "nowhere" }', "plugin torch"),
                ('{ kind = "torch", backnd = "gloo" }', "plugin torch has no backnd"),
                ('{ backend = "gloo" }', "names its kind"),
            ],
        )
        def it_refuses_one_that_is_not_there(tmp_path: Path, plugin: str, said: str) -> None:
            with pytest.raises(SystemExit, match=said):
                read(header(tmp_path, f'provider = "aws"\nplugins = [{plugin}]'))

    @pytest.mark.parametrize(
        ("text", "said"),
        [
            ("print('no header')\n", "declares no compute"),
            ("# /// script\n# dependencies = []\n# ///\n", "declares no compute"),
            ('# /// script\n# [tool.skyward]\n# provider = "aws"\n# ///\nimport os\n# /// script\n# dependencies = []\n# ///\n', "more than one"),
            ('# /// script\n# [tool.skyward]\n# provider = "aws"\n# nodez = 2\n# ///\n', "unknown field `nodez`"),
            ("# /// script\n# [tool.skyward]\n# provider = aws\n# ///\n", "Invalid value"),
        ],
    )
    def it_refuses_a_header_that_does_not_say_what_it_means(tmp_path: Path, text: str, said: str) -> None:
        with pytest.raises(SystemExit, match=said):
            read(written(tmp_path, text))


@pytest.mark.local
def describe_the_compute_a_script_is_named_after() -> None:
    def it_is_the_file_and_a_digest(tmp_path: Path) -> None:
        name = read(header(tmp_path, 'provider = "aws"')).name

        assert name.startswith("train-")
        assert len(name) == len("train-") + 8

    def the_same_header_is_the_same_compute(tmp_path: Path) -> None:
        first = read(header(tmp_path, 'provider = "aws"\naccelerator = "A100"', name="one.py")).name
        again = read(header(tmp_path, 'provider = "aws"\naccelerator = "A100"', name="one.py")).name

        assert first == again

    @pytest.mark.parametrize("change", ["nodes = 8", "nodes = { min = 1, max = 4 }", "delete_on_exit = false"])
    def what_a_compute_changes_in_place_is_not_part_of_it(tmp_path: Path, change: str) -> None:
        before = read(header(tmp_path, 'provider = "aws"', name="one.py")).name
        after = read(header(tmp_path, f'provider = "aws"\n{change}', name="one.py")).name

        assert before == after

    @pytest.mark.parametrize(
        ("skyward", "top"),
        [
            ('provider = "aws"\naccelerator = "H100"', ""),
            ('provider = "aws"', 'dependencies = ["torch"]'),
            ('provider = "aws"', 'requires-python = ">=3.13"'),
            ('provider = "aws"\nplugins = [{ kind = "torch" }]', ""),
            ('provider = "aws"\nimage = { apt = ["git"] }', ""),
            ('provider = "gcp"', ""),
        ],
    )
    def what_would_take_other_machines_is_another_compute(tmp_path: Path, skyward: str, top: str) -> None:
        before = read(header(tmp_path, 'provider = "aws"', name="one.py")).name
        after = read(header(tmp_path, skyward, top, name="one.py")).name

        assert before != after


@pytest.mark.local
def describe_running_the_text_on_a_node() -> None:
    def it_runs_as_main_with_the_argv_it_was_given(capsys: pytest.CaptureFixture[str]) -> None:
        source = "import sys\nif __name__ == '__main__':\n    print(sys.argv)\n"

        assert run(source, ("train.py", "--epochs", "3")) == 0
        assert capsys.readouterr().out == "['train.py', '--epochs', '3']\n"

    def it_gives_back_the_argv_it_found() -> None:
        import sys

        held = list(sys.argv)
        run("pass\n", ("train.py", "x"))

        assert sys.argv == held

    @pytest.mark.parametrize(("source", "status"), [("import sys; sys.exit(3)", 3), ("import sys; sys.exit()", 0), ("pass", 0)])
    def its_status_is_what_it_exited_with(source: str, status: int) -> None:
        assert run(source, ("train.py",)) == status

    def exiting_with_a_message_prints_it_and_is_a_failure(capsys: pytest.CaptureFixture[str]) -> None:
        assert run("import sys; sys.exit('out of data')", ("train.py",)) == 1
        assert capsys.readouterr().err == "out of data\n"

    def an_exception_is_printed_and_is_a_failure(capsys: pytest.CaptureFixture[str]) -> None:
        assert run("raise ValueError('bad shard')", ("train.py",)) == 1
        assert "ValueError: bad shard" in capsys.readouterr().err

    def a_stop_goes_through_to_the_worker_that_raised_it() -> None:
        with pytest.raises(Stop):
            run("from skyward.worker.stopping import Stop\nraise Stop\n", ("train.py",))

    def it_travels_by_reference_rather_than_as_bytecode() -> None:
        pickled = cloudpickle.dumps(partial(run, "print('hello')\n", ("train.py",)))

        assert b"skyward.worker.script" in pickled
        assert b"CodeType" not in pickled, "a code object is bytecode, and bytecode is only good on the interpreter that compiled it"


def declared(nodes: int, delete_on_exit: bool) -> str:
    """A script on local containers, that says where it ran and exits with the status it was handed."""
    return textwrap.dedent(f"""
        # /// script
        # requires-python = "=={PYTHON}.*"
        #
        # [tool.skyward]
        # provider = "container"
        # cpus = 1
        # memory_gb = 1
        # nodes = {nodes}
        # delete_on_exit = {str(delete_on_exit).lower()}
        #
        # [tool.skyward.image]
        # skyward = "local"
        # ///
        import sys

        import skyward as sky

        info = sky.instance_info()
        print(f"rank {{info.rank}} of {{info.nodes}} said {{sys.argv[1:]}}")
        sys.exit(int(sys.argv[1]))
    """).lstrip()


@pytest.mark.compute
@pytest.mark.xdist_group("script")
def describe_sky_run() -> None:
    def it_creates_the_compute_keeps_it_resizes_it_and_takes_it_down(daemon: str, tmp_path: Path) -> None:
        path = tmp_path / "lifecycle.py"
        path.write_text(declared(nodes=1, delete_on_exit=False))
        name = read(path).name
        try:
            first = cli("run", str(path), "--url", daemon, "--", "3")

            assert first.code == 3, first.err
            assert first.out.splitlines() == ["0 │ rank 0 of 1 said ['3']"], "stdout is the script's, each line after the rank that printed it"
            kept = httpx.get(f"{daemon}/v1/computes/{name}").json()
            assert kept["status"]["state"] == "ready", "delete_on_exit = false leaves the compute up"

            path.write_text(declared(nodes=2, delete_on_exit=False))
            second = cli("run", str(path), "--url", daemon, "--node", "all", "--", "0")

            assert second.code == 0, second.err
            broadcast = ["0 │ rank 0 of 2 said ['0']", "1 │ rank 1 of 2 said ['0']"]
            assert sorted(second.out.splitlines()) == broadcast, "a broadcast after a resize waits for the new size"
            (first_node,) = (node["id"] for node in kept["nodes"])
            assert first_node not in second.err, "a compute attached to is watched from now: the first node's bootstrap is not replayed"
            resized = httpx.get(f"{daemon}/v1/computes/{name}").json()
            assert resized["id"] == kept["id"], "the same header is the same compute"
            assert resized["generation"] == kept["generation"] + 1, "a new size is a resize, not another compute"

            path.write_text(declared(nodes=2, delete_on_exit=True))
            last = cli("run", str(path), "--url", daemon, "--node", "1", "--", "0")

            assert last.code == 0, last.err
            assert last.out.splitlines() == ["1 │ rank 1 of 2 said ['0']"]
            assert httpx.get(f"{daemon}/v1/computes/{name}").json()["status"]["state"] == "deleted"
        finally:
            if httpx.get(f"{daemon}/v1/computes/{name}").json().get("status", {}).get("state") not in (None, "deleted", "deleting"):
                cli("compute", "delete", name, "--url", daemon)
