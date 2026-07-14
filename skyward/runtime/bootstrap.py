from skyward.protocol.schemas import Image
from skyward.runtime.journal import SKYWARD_DIR

SCRIPT = f"{SKYWARD_DIR}/bootstrap.sh"
VENV = f"{SKYWARD_DIR}/.venv"
PYTHON = f"{VENV}/bin/python"

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


def script(image: Image, skyward: str) -> str:
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
    packages = " ".join(image.packages)
    exports = " ".join(f"export {key}={value!r};" for key, value in image.env.items())

    return "\n".join(
        (
            HEADER,
            f"phase env '{exports} true'" if exports else "",
            f"phase uv '{UV}'",
            f"phase venv 'uv venv {VENV} --python {python} --allow-existing'",
            f"phase skyward 'uv pip install --python {PYTHON} {skyward}'",
            f"phase deps 'uv pip install --python {PYTHON} {packages}'" if packages else "",
            FOOTER,
        ),
    )
