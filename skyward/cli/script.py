"""The compute a script declares, in the header ``uv run`` already reads.

PEP 723 puts a TOML block at the top of a file — ``requires-python`` and
``dependencies`` — and leaves ``[tool]`` to whoever else reads it. ``sky run``
reads the same block: the interpreter and the packages become the image the
nodes build, and ``[tool.skyward]`` is the rest of the compute, under the names
``sky.Compute`` gives the same things::

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

Where the script runs is not in the header. That is the command's to say, and it
may say something different on every run of the same file.
"""

from __future__ import annotations

import hashlib
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

import msgspec
from packaging.specifiers import InvalidSpecifier, SpecifierSet

from skyward.shared.schemas import Allocation, Image, NodeBounds, PipIndex, SkywardSource
from skyward.worker.plugins import PLUGINS, Plugin

PYTHONS = ("3.12", "3.13", "3.14")
"""The interpreters a node can be asked for, lowest first: ``requires-python`` gets the first it admits."""

BLOCK = re.compile(r"(?m)^# /// (?P<type>[a-zA-Z0-9-]+)$\s(?P<content>(^#(| .*)$\s)+)^# ///$")
"""PEP 723's own expression for a metadata block."""


@dataclass(frozen=True, slots=True)
class Script:
    """A file, and the compute its header asks for."""

    path: Path
    source: str
    provider: str
    accelerator: str | None
    cpus: int | None
    memory_gb: int | None
    region: str | None
    allocation: Allocation
    nodes: NodeBounds
    image: Image
    plugins: tuple[Plugin, ...]
    delete_on_exit: bool

    @property
    def name(self) -> str:
        """The compute's name: the file's, and a digest of everything that would take other machines.

        Two runs of one header land on one compute, and a header asking for another
        machine, image or plugin lands on another, because none of that changes on
        machines already bought. ``nodes`` is left out — a size is the one thing a
        compute changes in place, so a new one resizes this compute rather than
        naming a second — and so is ``delete_on_exit``, which is what a run does on
        its way out rather than what the machines are.
        """
        wanted = (
            self.provider,
            self.accelerator,
            self.cpus,
            self.memory_gb,
            self.region,
            self.allocation,
            self.image,
            [plugin.ref() for plugin in self.plugins],
        )
        digest = hashlib.sha256(msgspec.json.encode(wanted, order="sorted")).hexdigest()
        return f"{self.path.stem}-{digest[:8]}"


def read(path: Path) -> Script:
    """The script at ``path``, with the compute its header declares.

    Everything is checked here, before a daemon is asked for anything: a header
    that does not parse, names a field nobody has, or asks for a plugin that does
    not exist is refused with a sentence rather than discovered on a machine that
    is already billing.
    """
    source = path.read_text()
    match _metadata(path, source):
        case _Metadata(tool=_Tool(skyward=_Skyward() as declared)) as metadata:
            pass
        case _:
            raise SystemExit(f"{path} declares no compute: sky run reads the [tool.skyward] table of a `# /// script` block")

    return Script(
        path=path,
        source=source,
        provider=declared.provider,
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
            skyward=declared.image.skyward,
        ),
        plugins=tuple(_plugin(table) for table in declared.plugins),
        delete_on_exit=declared.delete_on_exit,
    )


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


__all__ = ["PYTHONS", "Script", "read"]
