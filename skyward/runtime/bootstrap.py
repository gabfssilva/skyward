import re
import shlex

from skyward.plugins.plugin import Plugin
from skyward.protocol.schemas import Image, MetricSpec
from skyward.runtime.journal import SKYWARD_DIR

SCRIPT = f"{SKYWARD_DIR}/bootstrap.sh"
VENV = f"{SKYWARD_DIR}/.venv"
PYTHON = f"{VENV}/bin/python"
VARS = f"{SKYWARD_DIR}/vars.sh"

HEADER = """#!/bin/bash
set -e

mkdir -p /opt/skyward
rm -f /opt/skyward/events.jsonl /opt/skyward/events.lock

export DEBIAN_FRONTEND=noninteractive
export UV_NO_PROGRESS=1
export PATH="/root/.local/bin:$PATH"

emit() {
    (
        flock 9
        printf '%s\\n' "$1" >> /opt/skyward/events.jsonl
    ) 9>/opt/skyward/events.lock
}

emit_phase() {
    emit "{\\"type\\":\\"phase\\",\\"event\\":\\"$1\\",\\"phase\\":\\"$2\\",\\"error\\":${3:-null}}"
}

emit_console() {
    local content="$1"
    content="${content//\\\\/\\\\\\\\}"
    content="${content//\\"/\\\\\\"}"
    content="${content//$'\\t'/\\\\t}"
    content="${content//$'\\r'/}"
    emit "{\\"type\\":\\"console\\",\\"content\\":\\"$content\\"}"
}

(echo 'set -e'; declare -f emit emit_console) > /opt/skyward/emit.sh

phase() {
    local name="$1"; shift
    emit_phase started "$name"

    set +e
    bash -c "$*" 2>&1 | while IFS= read -r line; do [ -n "$line" ] && emit_console "$line"; done
    local code=${PIPESTATUS[0]}
    set -e

    if [ "$code" -ne 0 ]; then
        emit_phase failed "$name" "\\"exit code $code\\""
        exit "$code"
    fi
    emit_phase completed "$name"
}

trap 'emit_phase failed bootstrap "\\"$BASH_COMMAND\\""' ERR

emit_phase started bootstrap
"""

FOOTER = "emit_phase completed bootstrap\n"

UV = "command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh"

_COLLECTORS: tuple[tuple[str, str, float], ...] = (
    ("cpu", "top -bn2 -d0.1 2>/dev/null | awk '/^%Cpu/{c=100-$8} END{printf \"%.1f\",c}'", 2),
    ("mem_used_mb", "free 2>/dev/null | awk '/^Mem:/{printf \"%d\",$3/1024}'", 2),
    ("mem_total_mb", "free 2>/dev/null | awk '/^Mem:/{printf \"%d\",$2/1024}'", 60),
    ("gpu_util", "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1;n++} END{if(n)printf \"%.1f\",s/n}'", 3),
    ("gpu_mem_mb", "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{if(NR)printf \"%d\",s}'", 3),
    ("gpu_mem_total_mb", "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{if(NR)printf \"%d\",s}'", 60),
)
"""What each node reports about itself, and how often. The ``gpu_*`` lines aggregate
across every GPU — utilisation averaged, memory summed — into one reading per node;
``nvidia-smi`` failing on a machine with no GPU leaves the reading empty and the
collector emits nothing.

The four the console shows — gpu, vram, cpu, mem — come out of these six: vram and
mem as used/total pairs, gpu and cpu as their own percentages."""


def _collector(name: str, command: str, interval: float) -> str:
    """One metric as a background shell loop: read, emit if numeric, sleep, repeat.

    ``set +e`` because a sample is allowed to fail — a busy ``nvidia-smi``, a missing
    interface — without taking the loop down with it; the next tick tries again. The
    regex is the gate: only a bare number reaches ``emit_metric``, so a command that
    printed a warning where a value should have been is dropped, not written as one.
    """
    return "\n".join(
        (
            f"_collect_{name}() {{",
            "    set +e",
            "    while true; do",
            f"        v=$({command})",
            f'        [[ "$v" =~ ^-?[0-9]*\\.?[0-9]+$ ]] && emit_metric {name} "$v"',
            f"        sleep {interval}",
            "    done",
            "}",
        ),
    )


def metrics(specs: tuple[MetricSpec, ...] | None = None) -> str:
    """The collectors, and the one call that sets them going.

    Started before the bootstrap phases and left running as background jobs: the
    script is ``nohup``-ed and non-interactive, so the loops outlive it and keep
    reporting while the worker runs — and still report if the worker never does.

    ``specs`` replaces the built-in six outright; ``None`` leaves them in place.
    """
    collectors = _COLLECTORS if specs is None else tuple((spec.name, spec.command, spec.interval) for spec in specs)
    return "\n".join(
        (
            'emit_metric() { emit "{\\"type\\":\\"metric\\",\\"name\\":\\"$1\\",\\"value\\":$2}"; }',
            *(_collector(name, command, interval) for name, command, interval in collectors),
            "start_metrics_daemon() {",
            *(f"    _collect_{name} &" for name, _, _ in collectors),
            "}",
            "start_metrics_daemon",
        ),
    )


def phase(name: str, *commands: str) -> str:
    """One bootstrap phase, quoted so its commands survive the ``bash -c`` the runner wraps them in.

    The commands are joined with ``&&`` — a phase fails on the first that does — and
    the whole run is a single shell word, which is what a plugin returns rather than
    reaching for the emit helpers itself.
    """
    return f"phase {name} {shlex.quote(' && '.join(commands))}"


def _package_name(spec: str) -> str:
    """The bare name a requirement scopes to, dropping any version or extras."""
    return re.split(r"[<>=!~ \[]", spec, maxsplit=1)[0]


def _apt(extra: tuple[str, ...]) -> str:
    """Install ``curl`` and ``git`` — the uv installer and git sources need them — plus extras.

    Always present: a minimal base image (RunPod) ships without ``curl``, so the
    fallback that installs uv is ``curl: command not found`` and the machine never
    gets a venv.
    """
    packages = " ".join(("curl", "git", *extra))
    return f"phase apt 'apt-get update -qq && apt-get install -y -qq {packages}'"


def _shell_vars(variables: dict[str, str]) -> str:
    """Resolve each command once and leave the results where later phases can read them.

    Every phase runs in its own ``bash -c`` subshell, so an export in one is gone by
    the next. The resolved values are written to ``vars.sh`` instead, and the phases
    that install packages source it — that is what carries a captured CUDA version
    from where it is read to the pip spec that needs it.
    """
    lines = [f": > {VARS}"]
    for name, command in variables.items():
        lines.append(f"{name}=$({command})")
        lines.append(f'printf "export {name}=%q\\n" "${name}" >> {VARS}')
    return f"phase shell_vars '{'; '.join(lines)}'"


def _pyproject(image: Image) -> str:
    """A project file whose only job is to scope indexes to the packages they own.

    ``uv pip install`` ignores ``[tool.uv.sources]``; only project resolution honours
    it. So when there are scoped indexes the install runs in project mode against this
    file, and ``explicit = true`` is the guarantee — a private index answers for the
    packages named against it and for nothing else.

    ``package = false`` keeps it a virtual project: uv manages ``.venv`` from it but
    does not try to build the root, which has no build backend and is not a package.
    """
    python = image.python or "3.13"
    lines = [
        "[project]",
        'name = "skyward-bootstrap"',
        'version = "0.0.0"',
        f'requires-python = ">={python}"',
        "",
        "[tool.uv]",
        "package = false",
        "",
    ]
    sources: dict[str, str] = {}
    for position, index in enumerate(image.pip_indexes):
        name = f"index-{position}"
        lines += ["[[tool.uv.index]]", f'name = "{name}"', f'url = "{index.url}"', "explicit = true", ""]
        for package in index.packages:
            sources[_package_name(package)] = name
    lines.append("[tool.uv.sources]")
    lines += [f'{package} = {{ index = "{name}" }}' for package, name in sources.items()]
    return "\n".join(lines)


def script(image: Image, skyward: str, plugins: tuple[Plugin, ...] = (), concurrency: int = 1) -> str:
    """The bootstrap, as a shell script the machine runs on its own.

    It is written to run **detached**, and to say what happened by appending to
    ``events.jsonl`` rather than by exiting with a code. That looks like a
    detour, and it is the whole point: the link to the machine can drop halfway
    through, and a bootstrap whose only record of itself was the exit status of
    an SSH command would be lost with it. The file survives, and the reader picks
    up from the line it got to.

    Parameters
    ----------
    image : Image
        The Python to install, the packages to put beside it, the environment to
        put them in.
    skyward : str
        What to install skyward from — a package name, a git URL, or the path of
        a wheel already uploaded to the machine. Which of the three it is has no
        bearing here, which is the reason a locally-built wheel needs no second
        script and no second pass: a failure to install skyward arrives as a
        failed phase in ``events.jsonl``, like every other failure.
    plugins : tuple[Plugin, ...]
        The compute's plugins, each asked for the phases it wants appended after
        the image's own bootstrap and before the footer.
    concurrency : int
        The worker's width, handed to each plugin's ``bootstrap`` — the datum a
        phase that partitions the machine needs and the image does not carry.

    Returns
    -------
    str
        A bash script. Its own emit helpers are left behind in ``emit.sh``, so
        that what runs after bootstrap — the worker — writes to the same file.

    Notes
    -----
    The venv is created with ``--allow-existing`` because a warm image already
    has one, holding the heavy wheels the image was baked to avoid downloading.
    Clearing it would throw away the only thing that made the image worth having.
    """
    python = image.python or "3.13"
    packages = " ".join(image.pip)
    exports = " ".join(f"export {key}={value!r};" for key, value in image.env.items())
    preload = f"source {VARS} 2>/dev/null; " if image.shell_vars else ""

    if image.pip_indexes:
        pyproject = f"cat > {SKYWARD_DIR}/pyproject.toml <<'PYPROJECT'\n{_pyproject(image)}\nPYPROJECT"
        install = f"cd {SKYWARD_DIR} && uv add --python {PYTHON}"
    else:
        pyproject = ""
        install = f"uv pip install --python {PYTHON}"

    postamble = "\n".join(op for plugin in plugins for op in plugin.bootstrap(image, concurrency))

    return "\n".join(
        (
            HEADER,
            metrics(image.metrics),
            _shell_vars(image.shell_vars) if image.shell_vars else "",
            f"phase env '{exports} true'" if exports else "",
            _apt(image.apt),
            f"phase uv '{UV}'",
            f"phase venv 'uv venv {VENV} --python {python} --allow-existing'",
            pyproject,
            f"phase skyward '{preload}{install} {skyward}'",
            f"phase deps '{preload}{install} {packages}'" if packages else "",
            *((postamble,) if postamble else ()),
            FOOTER,
        ),
    )
