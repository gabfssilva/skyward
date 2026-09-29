"""The compute a file declares, and what ``sky run`` runs on it.

A file declares its compute one of two ways. The first is the header ``uv run``
already reads. PEP 723 puts a TOML block at the top of a file — ``requires-python``
and ``dependencies`` — and leaves ``[tool]`` to whoever else reads it. ``sky run``
reads the same block: the interpreter and the packages become the image the nodes
build, ``[tool.skyward]`` is the rest of the compute, under the names
``sky.Compute`` gives the same things, and the whole file runs on the node::

    # /// script
    # requires-python = ">=3.12"
    # dependencies = ["safetensors"]
    #
    # [tool.skyward]
    # provider = "aws"
    # accelerator = "A100"
    #
    # [tool.skyward.nodes]
    # initial = 4
    # min = 2
    #
    # [[tool.skyward.plugins]]
    # kind = "torch"
    # ///

The second is ``sky.app``. The functions it decorates are the file's commands, and
each runs on the compute its ``sky.app`` describes::

    gpu = sky.app(provider=sky.AWS(), accelerator=sky.accelerators.A100())

    @gpu
    def train(epochs: int = 10) -> float: ...

    $ sky run train.py --epochs 5          # the file's one function
    $ sky run train.py train --epochs 5    # by name, as it has to be once there are two

The file is imported here to find them, and the command line is parsed against the
function's signature, so a value that does not fit is refused before a machine is
bought. A header is read without running anything, so a file with a
``[tool.skyward]`` table is a script whatever else it holds.

What the image includes is counted from the file, packed here, and sent with every
run beside the text, rather than unpacked once when a machine is set up: a run that
attaches to a compute left up gets the code as it is now.

Where the file runs is not in it. That is the command's to say, and it may say
something different on every run of the same file.
"""

from __future__ import annotations

import hashlib
import inspect
import re
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath

import cyclopts
import msgspec
from packaging.specifiers import InvalidSpecifier, SpecifierSet

from skyward.core import provider as factories
from skyward.core import usercode
from skyward.core.app import App, Entry
from skyward.core.spec import canonical
from skyward.shared.providers import Provider, split
from skyward.shared.schemas import Allocation, Image, NodeBounds, PipIndex, SkywardSource
from skyward.worker.plugins import PLUGINS, Plugin
from skyward.worker.script import loaded

PYTHONS = ("3.12", "3.13", "3.14")
"""The interpreters a node can be asked for, lowest first: ``requires-python`` gets the first it admits."""

BLOCK = re.compile(r"(?m)^# /// (?P<type>[a-zA-Z0-9-]+)$\s(?P<content>(^#(| .*)$\s)+)^# ///$")
"""PEP 723's own expression for a metadata block."""

RESERVED = frozenset({"node", "url"})
"""The flags ``sky run`` takes for itself, wherever they are on the line: a parameter by one of these names could never be given."""

FACTORIES: dict[str, Callable[[], Provider]] = {
    "aws": factories.AWS,
    "container": factories.Container,
    "fake": factories.Fake,
    "gcp": factories.GCP,
    "hyperstack": factories.Hyperstack,
    "jarvislabs": factories.JarvisLabs,
    "lambda": factories.Lambda,
    "massed_compute": factories.MassedCompute,
    "novita": factories.Novita,
    "runpod": factories.RunPod,
    "salad": factories.Salad,
    "scaleway": factories.Scaleway,
    "tensordock": factories.TensorDock,
    "vastai": factories.VastAI,
    "verda": factories.Verda,
    "vultr": factories.Vultr,
}


@dataclass(frozen=True, slots=True)
class Whole:
    """The file itself, run as ``__main__`` with ``argv``."""

    argv: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Call:
    """One of the file's functions, by the name the file binds it to, and its arguments as msgpack."""

    entry: str
    arguments: bytes


type Work = Whole | Call


@dataclass(frozen=True, slots=True)
class Script:
    """A file, the compute it declares, and what runs on it."""

    path: Path
    source: str
    app: App
    work: Work
    includes: bytes | None = None
    """What the image includes, packed the way the node unpacks it; ``None`` when it includes nothing."""

    @property
    def name(self) -> str:
        """The compute's name: the file's, and a digest of everything that would take other machines.

        Two runs of one declaration land on one compute, and one asking for another
        machine, image or plugin lands on another, because none of that changes on
        machines already bought. ``nodes`` is left out — a size is the one thing a
        compute changes in place, so a new one resizes this compute rather than
        naming a second — and so is ``delete_on_exit``, which is what a run does on
        its way out rather than what the machines are.

        A ``sky.app``'s accelerator counts by what it resolves to, so ``"A100"`` and
        ``sky.accelerators.A100()`` are one compute. A header's counts as written, as
        it has since headers first named computes: a name is only ever recomputed,
        never stored, so a header whose name moved would lose the machines it left up
        and buy others.
        """
        app = self.app
        match self.work:
            case Whole():
                accelerator = app.accelerator
            case Call():
                accelerator = canonical(app.accelerator)
        wanted = (
            _account(app.provider),
            accelerator,
            app.cpus,
            app.memory_gb,
            app.region,
            app.allocation,
            app.image,
            [plugin.ref() for plugin in app.plugins],
        )
        digest = hashlib.sha256(msgspec.json.encode(wanted, order="sorted")).hexdigest()
        return f"{self.path.stem}-{digest[:8]}"


def read(path: Path, tokens: Sequence[str] = ()) -> Script:
    """What ``sky run path tokens...`` runs: the whole file if its header declares the compute, else one of its ``sky.app`` functions.

    Everything is checked here, before a daemon is asked for anything: a header
    that does not parse, names a field nobody has, or asks for a plugin that does
    not exist, a command line the function cannot take, and an include that is not
    there, are refused with a sentence rather than discovered on a machine that is
    already billing.
    """
    source = path.read_text()
    match _metadata(path, source):
        case _Metadata(tool=_Tool(skyward=_Skyward() as declared)) as metadata:
            app, work = _declared(metadata, declared), Whole((str(path), *tokens))
        case _:
            app, work = _called(path, source, tokens)
    return Script(path, source, app, work, _packed(path, app.image))


def _account(provider: Provider) -> str | tuple[str, Mapping[str, object]]:
    """The provider as a compute's name counts it: its kind, and what was set apart from the kind's defaults.

    A default is not part of it, so a release that changes one renames nothing —
    a header names only a kind, and so its provider is only ever that. Nor is a
    credential: a key rotated buys nothing new.
    """
    _, settings = split(provider)
    _, defaults = split(type(provider)())
    changed = {key: value for key, value in settings.items() if value != defaults.get(key)}
    return (provider.kind, changed) if changed else provider.kind


def _declared(metadata: _Metadata, declared: _Skyward) -> App:
    if declared.provider not in FACTORIES:
        raise SystemExit(f"unknown provider '{declared.provider}'; known: {', '.join(sorted(FACTORIES))}")
    return App(
        provider=FACTORIES[declared.provider](),
        accelerator=declared.accelerator,
        cpus=declared.cpus,
        memory_gb=declared.memory_gb,
        region=declared.region,
        allocation=declared.allocation,
        nodes=_bounds(declared.nodes),
        image=Image(
            base=declared.image.base,
            python=_python(metadata.requires_python),
            pip=metadata.dependencies,
            apt=declared.image.apt,
            pip_indexes=declared.image.pip_indexes,
            env=declared.image.env,
            includes=declared.image.includes,
            excludes=declared.image.excludes,
            skyward=declared.image.skyward,
        ),
        plugins=tuple(_plugin(table) for table in declared.plugins),
        delete_on_exit=declared.delete_on_exit,
    )


def _called(path: Path, source: str, tokens: Sequence[str]) -> tuple[App, Call]:
    """The ``sky.app`` function ``tokens`` name, and the arguments they give it.

    With one function in the file, naming it is optional; with more, the name is
    the first token, and leaving it out is refused with the names there are. A
    function the file imports is not one of its commands — it belongs to the file
    that defined it. ``--help`` is the file's commands, or the one it names.
    """
    with loaded(source, str(path)) as module:
        entries = {name: value for name, value in vars(module).items() if isinstance(value, Entry) and value.fn.__module__ == module.__name__}
        if not entries:
            raise SystemExit(f"{path} declares no compute: sky run reads the [tool.skyward] table of a `# /// script` block, or the functions of a sky.app")

        parser = cyclopts.App(name=f"sky run {path.name}", version_flags=[])
        for name, entry in entries.items():
            if taken := sorted(RESERVED & inspect.signature(entry.fn).parameters.keys()):
                raise SystemExit(f"{name} takes {', '.join(taken)}, which sky run keeps for its own --{', --'.join(taken)}; rename it")
            parser.command(entry.fn, name=cyclopts.default_name_transform(name))
        if len(entries) == 1:
            (only,) = entries.values()
            parser.default(only.fn)

        chosen, bound, _ = parser.parse_args(list(tokens))
        if (name := next((name for name, entry in entries.items() if entry.fn is chosen), None)) is None:
            chosen(*bound.args, **bound.kwargs)
            raise SystemExit(0)
        return entries[name].app, Call(name, _encoded(bound.arguments))


def _packed(path: Path, image: Image) -> bytes | None:
    """What ``image`` includes, as the tar.gz the node unpacks onto ``sys.path``.

    Each path counts from the file rather than from wherever ``sky run`` was typed,
    so the file ships the same code from anywhere, and lands under its own name:
    ``src/classy_enc`` is the package ``classy_enc``. Two paths landing under one
    name would be merged into one directory, so they are refused, as is one that is
    not there.
    """
    if not image.includes:
        return None
    found = [(path.parent / include).resolve() for include in image.includes]
    if missing := [include for include, place in zip(image.includes, found, strict=True) if not place.exists()]:
        raise SystemExit(f"{path} includes what is not there: {', '.join(missing)}")
    names = [place.name for place in found]
    if doubled := sorted({name for name in names if names.count(name) > 1}):
        raise SystemExit(f"{path} includes more than one path named {', '.join(doubled)}")
    return usercode.tarball([str(place) for place in found], image.excludes)


def _encoded(arguments: Mapping[str, object]) -> bytes:
    """The parsed arguments as msgpack, a path as its text: :func:`skyward.worker.script.call` converts them back."""
    try:
        return msgspec.msgpack.encode(dict(arguments), enc_hook=_text)
    except TypeError as unsupported:
        raise SystemExit(f"an argument cannot travel to the node: {unsupported}") from None


def _text(value: object) -> object:
    if isinstance(value, PurePath):
        return str(value)
    raise NotImplementedError(type(value))


class _Nodes(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    initial: int | None = None
    min: int | None = None
    max: int | None = None


class _Image(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """The image's fields a header may set: the interpreter and the packages come from PEP 723's own keys."""

    base: str | None = None
    apt: tuple[str, ...] = ()
    env: dict[str, str] = {}
    pip_indexes: tuple[PipIndex, ...] = ()
    includes: tuple[str, ...] = ()
    excludes: tuple[str, ...] = ()
    skyward: SkywardSource = "auto"


class _Skyward(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    provider: str
    accelerator: str | None = None
    cpus: int | None = None
    memory_gb: int | None = None
    region: str | None = None
    allocation: Allocation = "spot_if_available"
    nodes: int | _Nodes = 1
    image: _Image = msgspec.field(default_factory=_Image)
    plugins: tuple[dict[str, object], ...] = ()
    delete_on_exit: bool = True


class _Tool(msgspec.Struct, frozen=True):
    skyward: _Skyward | None = None


class _Metadata(msgspec.Struct, frozen=True):
    requires_python: str | None = msgspec.field(default=None, name="requires-python")
    dependencies: tuple[str, ...] = ()
    tool: _Tool = msgspec.field(default_factory=_Tool)


def _metadata(path: Path, source: str) -> _Metadata | None:
    blocks = [block for block in BLOCK.finditer(source) if block.group("type") == "script"]
    match blocks:
        case []:
            return None
        case [block]:
            toml = "".join(line[2:] if line.startswith("# ") else line[1:] for line in block.group("content").splitlines(keepends=True))
            try:
                return msgspec.convert(tomllib.loads(toml), _Metadata)
            except (tomllib.TOMLDecodeError, msgspec.ValidationError) as invalid:
                raise SystemExit(f"{path}: {invalid}") from None
        case _:
            raise SystemExit(f"{path} has more than one `# /// script` block")


def _python(specifier: str | None) -> str | None:
    """The first interpreter a node can run that ``requires-python`` admits."""
    if specifier is None:
        return None
    try:
        admitted = SpecifierSet(specifier)
    except InvalidSpecifier:
        raise SystemExit(f"requires-python is not a version specifier: {specifier!r}") from None
    if (chosen := next((version for version in PYTHONS if version in admitted), None)) is None:
        raise SystemExit(f"requires-python {specifier!r} admits none of {', '.join(PYTHONS)}")
    return chosen


def _bounds(nodes: int | _Nodes) -> NodeBounds:
    """``nodes = 4``, or a table of ``initial``, ``min`` and ``max`` — ``sky.Nodes`` in TOML.

    A table without ``initial`` opens at its floor, the way ``nodes=(min, max)``
    does in the SDK.
    """
    match nodes:
        case int(count) if count >= 1:
            return NodeBounds(initial=count)
        case _Nodes(initial=int(opening), min=floor, max=ceiling) | _Nodes(initial=None, min=int(opening) as floor, max=ceiling):
            lowest = opening if floor is None else floor
            highest = opening if ceiling is None else ceiling
            if not 0 <= lowest <= opening <= highest or highest < 1:
                given = f"initial={opening}, min={floor}, max={ceiling}"
                raise SystemExit(f"[tool.skyward.nodes] needs 0 <= min <= initial <= max, and room for one node; got {given}")
            return NodeBounds(initial=opening, min=floor, max=ceiling)
        case _Nodes():
            raise SystemExit("[tool.skyward.nodes] needs an initial or a min")
        case _:
            raise SystemExit(f"nodes takes a count of at least one, or a table; not {nodes}")


def _plugin(table: dict[str, object]) -> Plugin:
    """One ``[[tool.skyward.plugins]]`` entry: its ``kind``, and the plugin's own fields beside it."""
    given = {key: value for key, value in table.items() if key != "kind"}
    match table.get("kind"):
        case str(kind) if kind in PLUGINS:
            plugin = PLUGINS[kind]
            if unknown := given.keys() - {field.name for field in msgspec.structs.fields(plugin)}:
                raise SystemExit(f"plugin {kind} has no {', '.join(sorted(unknown))}")
            try:
                return msgspec.convert(given, plugin)
            except msgspec.ValidationError as invalid:
                raise SystemExit(f"plugin {kind}: {invalid}") from None
        case str(kind):
            raise SystemExit(f"unknown plugin '{kind}'; known: {', '.join(sorted(PLUGINS))}")
        case _:
            raise SystemExit("every [[tool.skyward.plugins]] entry names its kind")


__all__ = ["FACTORIES", "PYTHONS", "Call", "Script", "Whole", "Work", "read"]
