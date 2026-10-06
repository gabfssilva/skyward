"""A file that declares its own compute, and the node that runs it.

``sky run`` reads the PEP 723 block at the top of a file — ``requires-python`` and
``dependencies`` are the image, ``[tool.skyward]`` the rest of the compute — or the
functions a ``sky.app`` decorates. What is asserted here is what either comes to
before any daemon is asked for anything, what the node does with the text once it
has it, and — on containers — what a run does with the compute: create it, keep
it, resize it and take it down.
"""

from __future__ import annotations

import io
import json
import sys
import tarfile
import textwrap
import types
from functools import partial
from pathlib import Path

import cloudpickle
import httpx
import msgspec
import pytest

from skyward.cli.script import Call, Script, Whole, read
from skyward.core import usercode
from skyward.core.accelerators import Accelerator
from skyward.core.app import App
from skyward.core.spec import Options
from skyward.shared.providers import AWS
from skyward.shared.schemas import NodeBounds, PipIndex
from skyward.worker.plugins import HuggingFace, Torch
from skyward.worker.script import Exited, Returned, call, run
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


def shipped(root: Path, body: str = "VERSION = 1\n") -> Path:
    """A package named ``shipped`` under ``root``, whose ``__init__`` holds ``body``; answers with that file."""
    (root / "shipped").mkdir(parents=True, exist_ok=True)
    init = root / "shipped" / "__init__.py"
    init.write_text(body)
    return init


def archive(root: Path) -> bytes:
    """The package under ``root``, packed as ``sky run`` sends it."""
    return usercode.tarball([str(root / "shipped")])


def members(script: Script) -> list[str]:
    assert script.includes is not None
    with tarfile.open(fileobj=io.BytesIO(script.includes)) as packed:
        return sorted(packed.getnames())


@pytest.mark.local
def describe_reading_a_header() -> None:
    def it_is_the_compute_sky_compute_would_be_given(tmp_path: Path) -> None:
        script = read(written(tmp_path, HEADER))
        app = script.app

        assert (app.provider, app.accelerator, app.allocation) == (AWS(), "A100", "spot")
        assert app.nodes == NodeBounds(initial=4, min=2)
        assert app.image.python == "3.12"
        assert tuple(app.image.pip) == ("safetensors", "numpy<3")
        assert tuple(app.image.apt) == ("htop",)
        assert app.image.env == {"NCCL_DEBUG": "WARN"}
        assert tuple(app.image.pip_indexes) == (PipIndex(url="https://download.pytorch.org/whl/cu128", packages=("torch",)),)
        assert app.plugins == (Torch(backend="gloo"), HuggingFace())
        assert app.delete_on_exit is True
        assert script.source.endswith('print("hello")\n')

    def the_whole_file_runs_with_what_follows_it_as_its_argv(tmp_path: Path) -> None:
        path = written(tmp_path, HEADER)

        assert read(path, ["--epochs", "3"]).work == Whole((str(path), "--epochs", "3"))

    def it_is_read_without_running_the_file(tmp_path: Path) -> None:
        path = header(tmp_path, 'provider = "aws"')
        path.write_text(path.read_text().replace("print('hello')", "raise RuntimeError('ran')"))

        assert read(path).work == Whole((str(path),))

    def describe_requires_python() -> None:
        @pytest.mark.parametrize(
            ("specifier", "chosen"),
            [(">=3.12", "3.12"), (">=3.13", "3.13"), ("==3.14.*", "3.14"), ("~=3.12", "3.12"), (">=3.12.4", "3.13")],
        )
        def it_is_the_first_interpreter_it_admits(tmp_path: Path, specifier: str, chosen: str) -> None:
            assert read(header(tmp_path, 'provider = "aws"', f'requires-python = "{specifier}"')).app.image.python == chosen

        def left_out_it_is_the_image_default(tmp_path: Path) -> None:
            assert read(header(tmp_path, 'provider = "aws"')).app.image.python is None

        @pytest.mark.parametrize(("specifier", "said"), [("<3.12", "admits none of"), ("three", "not a version specifier")])
        def it_refuses_one_no_node_can_satisfy(tmp_path: Path, specifier: str, said: str) -> None:
            with pytest.raises(SystemExit, match=said):
                read(header(tmp_path, 'provider = "aws"', f'requires-python = "{specifier}"'))

    def describe_nodes() -> None:
        def a_count_is_a_fixed_size(tmp_path: Path) -> None:
            assert read(header(tmp_path, 'provider = "aws"\nnodes = 3')).app.nodes == NodeBounds(initial=3)

        def a_table_without_initial_opens_at_its_floor(tmp_path: Path) -> None:
            assert read(header(tmp_path, 'provider = "aws"\nnodes = { min = 2, max = 8 }')).app.nodes == NodeBounds(initial=2, min=2, max=8)

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
            ('# /// script\n# [tool.skyward]\n# provider = "nowhere"\n# ///\n', "unknown provider 'nowhere'"),
        ],
    )
    def it_refuses_a_header_that_does_not_say_what_it_means(tmp_path: Path, text: str, said: str) -> None:
        with pytest.raises(SystemExit, match=said):
            read(written(tmp_path, text))


MODULE = """
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import skyward as sky


@dataclass(frozen=True)
class Optim:
    lr: float = 1e-3


gpu = sky.app(provider=sky.AWS(), accelerator=sky.accelerators.A100(), nodes=2, plugins=[sky.plugins.Torch()])


def helper() -> int:
    return 1


@gpu
def train(epochs: int = 10, data: Path = Path("data"), mode: Literal["fast", "slow"] = "fast", optim: Optim = Optim()) -> dict[str, object]:
    \"\"\"Train the model.\"\"\"
    return {"epochs": epochs, "data": data, "mode": mode, "lr": optim.lr}


if __name__ == "__main__":
    raise SystemExit("the main block ran")
"""

EVALUATE = """

@gpu
def evaluate(checkpoint: Path) -> float:
    return 0.5
"""


def arguments(script: Script) -> object:
    """What the command line was parsed into, as the node will receive it."""
    match script.work:
        case Call(arguments=encoded):
            return msgspec.msgpack.decode(encoded)
        case Whole():
            raise AssertionError("a sky.app file runs one of its functions, not the whole file")


def named(tmp_path: Path, app: App) -> str:
    """The compute ``app`` is, declared by a ``sky.app`` in a file called ``one.py``."""
    return Script(tmp_path / "one.py", "", app, Call("train", b"")).name


@pytest.mark.local
def describe_reading_a_module() -> None:
    def it_finds_the_function_and_the_compute_above_it(tmp_path: Path) -> None:
        script = read(written(tmp_path, MODULE), ["--epochs", "5", "--data", "/d", "--mode", "slow", "--optim.lr", "0.1"])

        assert (script.app.provider, script.app.accelerator, script.app.nodes) == (AWS(), Accelerator("a100"), 2)
        assert script.app.plugins == (Torch(),)
        assert isinstance(script.work, Call)
        assert script.work.entry == "train"
        assert arguments(script) == {"epochs": 5, "data": "/d", "mode": "slow", "optim": {"lr": 0.1}}

    def the_one_function_can_be_named_too(tmp_path: Path) -> None:
        assert arguments(read(written(tmp_path, MODULE), ["train", "--epochs", "5"])) == {"epochs": 5}

    def what_is_left_out_is_the_function_s_own_default(tmp_path: Path) -> None:
        assert arguments(read(written(tmp_path, MODULE))) == {}

    def the_main_block_stays_out(tmp_path: Path) -> None:
        assert read(written(tmp_path, MODULE)).source.endswith('raise SystemExit("the main block ran")\n')

    @pytest.mark.parametrize(("tokens", "said"), [(["--epochs", "five"], "--epochs"), (["--mode", "medium"], "--mode"), (["--nope", "1"], "--nope")])
    def it_refuses_a_command_line_the_function_cannot_take(tmp_path: Path, capsys: pytest.CaptureFixture[str], tokens: list[str], said: str) -> None:
        with pytest.raises(SystemExit) as refused:
            read(written(tmp_path, MODULE), tokens)

        assert refused.value.code == 1
        assert said in capsys.readouterr().err

    def help_lists_the_function_and_buys_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as helped:
            read(written(tmp_path, MODULE), ["--help"])

        assert helped.value.code == 0
        assert "--epochs" in capsys.readouterr().out

    def describe_with_two_functions() -> None:
        def a_command_is_named(tmp_path: Path) -> None:
            script = read(written(tmp_path, MODULE + EVALUATE), ["evaluate", "--checkpoint", "/c"])

            assert isinstance(script.work, Call)
            assert (script.work.entry, arguments(script)) == ("evaluate", {"checkpoint": "/c"})

        def leaving_the_name_out_is_refused_with_the_names_there_are(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
            with pytest.raises(SystemExit):
                read(written(tmp_path, MODULE + EVALUATE), ["--epochs", "5"])

            assert "train, evaluate" in capsys.readouterr().err

        def a_command_is_the_name_the_file_binds_it_to(tmp_path: Path) -> None:
            script = read(written(tmp_path, MODULE + EVALUATE + "\n\nscore_all = gpu(lambda: 1.0)\n"), ["score-all"])

            assert isinstance(script.work, Call)
            assert script.work.entry == "score_all"

    def a_function_the_file_imports_is_not_one_of_its_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "borrowed_entries.py").write_text(MODULE)
        monkeypatch.syspath_prepend(str(tmp_path))

        with pytest.raises(SystemExit, match="declares no compute"):
            read(written(tmp_path, "from borrowed_entries import train\n", name="main.py"))

    def it_refuses_a_parameter_sky_run_keeps_for_itself(tmp_path: Path) -> None:
        with pytest.raises(SystemExit, match="train takes node, which sky run keeps for its own --node"):
            read(written(tmp_path, "import skyward as sky\n\n@sky.app(provider=sky.AWS())\ndef train(node: int) -> None: ...\n"))

    def a_header_wins_over_what_the_file_decorates(tmp_path: Path) -> None:
        path = header(tmp_path, 'provider = "gcp"')
        path.write_text(path.read_text() + MODULE)

        assert read(path).work == Whole((str(path),))


@pytest.mark.local
def describe_what_the_image_includes() -> None:
    def it_counts_from_the_file_and_lands_under_its_own_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        project = tmp_path / "project"
        shipped(project / "src")
        path = written(
            project,
            """
            import skyward as sky


            @sky.app(provider=sky.AWS(), image=sky.Image(includes=["src/shipped"]))
            def version() -> int:
                return 1
            """,
        )
        monkeypatch.chdir(tmp_path)

        assert members(read(path)) == ["shipped/__init__.py"]

    def a_header_includes_under_its_image_table_and_leaves_out_what_it_excludes(tmp_path: Path) -> None:
        shipped(tmp_path / "src")
        (tmp_path / "src" / "shipped" / "rows.csv").write_text("1,2\n")

        script = read(header(tmp_path, 'provider = "aws"\nimage = { includes = ["src/shipped"], excludes = ["*.csv"] }'))

        assert (script.app.image.includes, script.app.image.excludes) == (("src/shipped",), ("*.csv",))
        assert members(script) == ["shipped/__init__.py"]

    def nothing_included_is_nothing_sent(tmp_path: Path) -> None:
        assert read(header(tmp_path, 'provider = "aws"')).includes is None

    def what_is_not_there_is_refused(tmp_path: Path) -> None:
        with pytest.raises(SystemExit, match="includes what is not there: src/gone"):
            read(header(tmp_path, 'provider = "aws"\nimage = { includes = ["src/gone"] }'))

    def two_paths_that_would_land_under_one_name_are_refused(tmp_path: Path) -> None:
        shipped(tmp_path / "a")
        shipped(tmp_path / "b")

        with pytest.raises(SystemExit, match="more than one path named shipped"):
            read(header(tmp_path, 'provider = "aws"\nimage = { includes = ["a/shipped", "b/shipped"] }'))


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

    @pytest.mark.parametrize(
        ("skyward", "name"),
        [
            ('provider = "aws"', "dp_perf-4deb0c66"),
            ('provider = "salad"\naccelerator = "RTX_3090"\nallocation = "on_demand"', "dp_perf-98159751"),
        ],
    )
    def a_header_keeps_the_name_it_has_always_had(tmp_path: Path, skyward: str, name: str) -> None:
        """A name is recomputed on every run and never stored: one that moved would lose the machines a header left up."""
        top = 'requires-python = ">=3.13"\ndependencies = ["torch==2.14.0", "numpy"]'

        assert read(header(tmp_path, skyward, top, name="dp_perf.py")).name == name

    def the_paths_the_image_includes_are_part_of_it_and_what_is_in_them_is_not(tmp_path: Path) -> None:
        version = shipped(tmp_path)
        bare = read(header(tmp_path, 'provider = "aws"', name="one.py")).name
        including = read(header(tmp_path, 'provider = "aws"\nimage = { includes = ["shipped"] }', name="one.py")).name
        version.write_text("VERSION = 2\n")
        edited = read(header(tmp_path, 'provider = "aws"\nimage = { includes = ["shipped"] }', name="one.py")).name

        assert bare != including
        assert including == edited, "an edit travels with the next run, to the compute already up"

    def two_functions_under_one_app_are_one_compute(tmp_path: Path) -> None:
        path = written(tmp_path, MODULE + EVALUATE)

        assert read(path, ["train"]).name == read(path, ["evaluate", "--checkpoint", "/c"]).name

    def describe_an_app() -> None:
        @pytest.mark.parametrize(
            "other",
            [
                App(provider=AWS(access_key_id="a", secret_access_key="b")),
                App(provider=AWS(region="us-east-1")),
                App(provider=AWS(), nodes=8),
                App(provider=AWS(), delete_on_exit=False),
                App(provider=AWS(), options=Options(ready_timeout=1800)),
            ],
        )
        def what_is_not_the_machines_is_not_part_of_it(tmp_path: Path, other: App) -> None:
            assert named(tmp_path, App(provider=AWS())) == named(tmp_path, other)

        @pytest.mark.parametrize("spelled", ["A100", "a100", Accelerator("a100")])
        def an_accelerator_is_what_it_resolves_to(tmp_path: Path, spelled: str | Accelerator) -> None:
            assert named(tmp_path, App(provider=AWS(), accelerator=spelled)) == named(tmp_path, App(provider=AWS(), accelerator="A100"))

        @pytest.mark.parametrize(
            "other",
            [
                App(provider=AWS(region="us-west-2")),
                App(provider=AWS(), accelerator="H100"),
                App(provider=AWS(), accelerator=Accelerator("a100", count=2)),
                App(provider=AWS(), plugins=[Torch()]),
            ],
        )
        def what_would_take_other_machines_is_another_compute(tmp_path: Path, other: App) -> None:
            assert named(tmp_path, App(provider=AWS())) != named(tmp_path, other)


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

    def what_the_image_includes_imports_by_its_own_name_while_it_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        shipped(tmp_path)
        held = list(sys.path)

        assert run("from shipped import VERSION\nprint(VERSION)\n", ("train.py",), archive(tmp_path)) == 0
        assert capsys.readouterr().out == "1\n"
        assert sys.path == held
        assert "shipped" not in sys.modules

    def each_run_imports_the_copy_it_was_sent(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        version = shipped(tmp_path)
        first = archive(tmp_path)
        version.write_text("VERSION = 2\n")
        second = archive(tmp_path)

        run("from shipped import VERSION\nprint(VERSION)\n", ("train.py",), first)
        run("from shipped import VERSION\nprint(VERSION)\n", ("train.py",), second)

        assert capsys.readouterr().out == "1\n2\n"

    def a_module_that_makes_up_any_attribute_leaves_what_it_includes_to_be_cleaned_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # torch.classes answers every name it is asked for, __path__ included
        class Anything(types.ModuleType):
            def __getattr__(self, name: str) -> object:
                return object()

        monkeypatch.setitem(sys.modules, "anything", Anything("anything"))
        shipped(tmp_path)

        assert run("import shipped\n", ("train.py",), archive(tmp_path)) == 0
        assert "shipped" not in sys.modules

    def a_traceback_through_what_it_includes_shows_the_lines(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        shipped(tmp_path, "def fail() -> None:\n    raise ValueError('from the package')\n")

        assert run("from shipped import fail\nfail()\n", ("train.py",), archive(tmp_path)) == 1
        assert "raise ValueError('from the package')" in capsys.readouterr().err


@pytest.mark.local
def describe_calling_a_function_on_a_node() -> None:
    def it_converts_the_arguments_back_and_answers_with_json() -> None:
        given = msgspec.msgpack.encode({"epochs": 3, "data": "/d", "mode": "slow", "optim": {"lr": 0.5}})

        assert call(MODULE, "train.py", "train", given) == Returned(b'{"epochs":3,"data":"/d","mode":"slow","lr":0.5}')

    def what_is_left_out_is_the_function_s_own_default() -> None:
        assert call(MODULE, "train.py", "train", msgspec.msgpack.encode({})) == Returned(b'{"epochs":10,"data":"data","mode":"fast","lr":0.001}')

    def star_args_and_star_star_kwargs_arrive_whole() -> None:
        source = entry("def gather(*paths: Path, **weights: float) -> list[object]:\n    return [[type(path).__name__ for path in paths], weights]")
        given = msgspec.msgpack.encode({"paths": ["/a", "/b"], "weights": {"x": 1}})

        assert call(source, "gather.py", "gather", given) == Returned(b'[["PosixPath","PosixPath"],{"x":1.0}]')

    def it_runs_under_the_file_s_name_not_main() -> None:
        assert call(entry("def name() -> str:\n    return __name__"), "/somewhere/train.py", "name", msgspec.msgpack.encode({})) == Returned(b'"train"')

    def it_leaves_sys_modules_as_it_found_it() -> None:
        call(MODULE, "leftover.py", "train", msgspec.msgpack.encode({}))

        assert "leftover" not in sys.modules

    def a_value_json_has_no_form_for_is_its_str() -> None:
        model = "class Model:\n    def __str__(self) -> str:\n        return 'a model'\n"
        source = entry("def fit() -> object:\n    return {'model': Model()}", above=model)

        assert call(source, "fit.py", "fit", msgspec.msgpack.encode({})) == Returned(b'{"model":"a model"}')

    @pytest.mark.parametrize(("body", "status"), [("sys.exit(3)", 3), ("sys.exit()", 0), ("raise ValueError('bad shard')", 1)])
    def exiting_and_raising_end_it_as_they_end_a_script(capsys: pytest.CaptureFixture[str], body: str, status: int) -> None:
        assert call(entry(f"def fail() -> None:\n    {body}"), "fail.py", "fail", msgspec.msgpack.encode({})) == Exited(status)
        assert status != 1 or "ValueError: bad shard" in capsys.readouterr().err

    def a_stop_goes_through_to_the_worker_that_raised_it() -> None:
        with pytest.raises(Stop):
            call(entry("def stop() -> None:\n    raise Stop"), "stop.py", "stop", msgspec.msgpack.encode({}))

    def it_travels_by_reference_rather_than_as_bytecode() -> None:
        pickled = cloudpickle.dumps(partial(call, MODULE, "train.py", "train", msgspec.msgpack.encode({})))

        assert b"skyward.worker.script" in pickled
        assert b"CodeType" not in pickled, "a code object is bytecode, and bytecode is only good on the interpreter that compiled it"

    def what_the_image_includes_is_there_for_the_function(tmp_path: Path) -> None:
        shipped(tmp_path)
        source = entry("def version() -> int:\n    from shipped import VERSION\n    return VERSION")

        assert call(source, "version.py", "version", msgspec.msgpack.encode({}), archive(tmp_path)) == Returned(b"1")
        assert "shipped" not in sys.modules


def entry(function: str, above: str = "") -> str:
    """A file whose one ``sky.app`` function is ``function``, below ``above``, with what the cases reach for imported."""
    imports = "import sys\nfrom pathlib import Path\n\nimport skyward as sky\nfrom skyward.worker.stopping import Stop\n"
    return f"{imports}\n{above}\n\n@sky.app(provider=sky.AWS())\n{function}\n"


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


def entries(nodes: int, delete_on_exit: bool) -> str:
    """A file on local containers whose functions say where they ran and what they were given."""
    return textwrap.dedent(f"""
        import skyward as sky

        local = sky.app(
            provider=sky.Container(),
            cpus=1,
            memory_gb=1,
            nodes={nodes},
            image=sky.Image(python="{PYTHON}", skyward="local"),
            delete_on_exit={delete_on_exit},
        )


        @local
        def shout(word: str, times: int = 1) -> dict[str, object]:
            rank = sky.instance_info().rank
            print(f"rank {{rank}} shouting")
            return {{"rank": rank, "said": word * times}}


        @local
        def fail() -> None:
            raise ValueError("on purpose")
    """).lstrip()


def including(delete_on_exit: bool) -> str:
    """A file on local containers whose function answers with what the package its image includes says."""
    return textwrap.dedent(f"""
        import skyward as sky


        @sky.app(
            provider=sky.Container(),
            cpus=1,
            memory_gb=1,
            image=sky.Image(python="{PYTHON}", skyward="local", includes=["src/shipped"]),
            delete_on_exit={delete_on_exit},
        )
        def version() -> int:
            from shipped import VERSION

            return VERSION
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

    def it_runs_a_function_and_prints_what_it_returned(daemon: str, tmp_path: Path) -> None:
        path = tmp_path / "shouting.py"
        path.write_text(entries(nodes=2, delete_on_exit=False))
        name = read(path, ["fail"]).name
        try:
            everywhere = cli("run", "--url", daemon, "--node", "all", str(path), "shout", "--word", "hi", "--times", "2")

            assert everywhere.code == 0, everywhere.err
            *lines, answer = everywhere.out.splitlines()
            assert sorted(lines) == ["0 │ rank 0 shouting", "1 │ rank 1 shouting"], "a node runs the file with no httpx, sky.app and all"
            assert json.loads(answer) == [{"rank": 0, "said": "hihi"}, {"rank": 1, "said": "hihi"}], "every node's answer, by rank, after the lines"

            path.write_text(entries(nodes=2, delete_on_exit=True))
            failed = cli("run", "--url", daemon, "--node", "1", str(path), "fail")

            assert failed.code == 1, failed.err
            assert "ValueError: on purpose" in failed.out
            assert all(line.startswith("1 │") for line in failed.out.splitlines()), "a function that raised answers nothing"
            assert httpx.get(f"{daemon}/v1/computes/{name}").json()["status"]["state"] == "deleted"
        finally:
            if httpx.get(f"{daemon}/v1/computes/{name}").json().get("status", {}).get("state") not in (None, "deleted", "deleting"):
                cli("compute", "delete", name, "--url", daemon)

    def what_the_image_includes_goes_with_every_run(daemon: str, tmp_path: Path) -> None:
        version = shipped(tmp_path / "src")
        path = tmp_path / "versioned.py"
        path.write_text(including(delete_on_exit=False))
        name = read(path).name
        try:
            first = cli("run", "--url", daemon, str(path))

            assert first.code == 0, first.err
            assert first.out.splitlines() == ["1"]
            kept = httpx.get(f"{daemon}/v1/computes/{name}").json()

            version.write_text("VERSION = 2\n")
            path.write_text(including(delete_on_exit=True))
            second = cli("run", "--url", daemon, str(path))

            assert second.code == 0, second.err
            assert second.out.splitlines() == ["2"], "a compute attached to runs the code as it is now, not as it was when it came up"
            attached = httpx.get(f"{daemon}/v1/computes/{name}").json()
            assert attached["id"] == kept["id"]
            assert attached["spec"]["image"]["includes"] == [], "the machines are built without what goes with every run"
        finally:
            if httpx.get(f"{daemon}/v1/computes/{name}").json().get("status", {}).get("state") not in (None, "deleted", "deleting"):
                cli("compute", "delete", name, "--url", daemon)
