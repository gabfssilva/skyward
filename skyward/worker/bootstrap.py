from __future__ import annotations

import hashlib
import re
import shlex
from collections.abc import Sequence
from typing import TYPE_CHECKING

from skyward.shared.schemas import READINGS, Endpoint, Image, MetricSpec, Reading, Volume
from skyward.worker.journal import SKYWARD_DIR

if TYPE_CHECKING:
    from skyward.worker.plugins.plugin import Plugin

SCRIPT = f"{SKYWARD_DIR}/bootstrap.sh"
VENV = f"{SKYWARD_DIR}/.venv"
PYTHON = f"{VENV}/bin/python"
VARS = f"{SKYWARD_DIR}/vars.sh"
ENV = f"{SKYWARD_DIR}/env.sh"

GEESEFS = "v0.43.8"
"""The geesefs release the nodes mount with, pinned.

An unpinned ``/releases/latest`` is a different filesystem driver on every boot,
which is a fleet that reads the same bucket two ways and a bug that cannot be
reproduced from the spec that caused it.
"""
GEESEFS_BIN = "/usr/local/bin/geesefs"
FUSE_ROOT = "/mnt/geesefs"

HEADER = """#!/bin/bash
set -e

mkdir -p /opt/skyward
rm -f /opt/skyward/events.jsonl /opt/skyward/events.lock

export DEBIAN_FRONTEND=noninteractive
export UV_NO_PROGRESS=1
export PATH="/root/.local/bin:$PATH"

emit() {
    { flock 9; printf '%s\\n' "$1" >> /opt/skyward/events.jsonl; } 9>/opt/skyward/events.lock
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

_BUILTIN = r'''_collect_builtin() {
    set +e
    local wants=" $* "
    local tick=0 gpu=0 now lines line pct busy total prev_busy=0 prev_total=0
    local cpu user nice system idle iowait irq softirq steal rest
    local key value unit mem_total mem_available
    local util used capacity temp power count util_sum used_sum capacity_sum temp_max power_sum powered cents
    local iface counters rx_sum tx_sum prev_rx=0 prev_tx=0 prev_net=0 elapsed rate
    local filesystem blocks used_blocks available capacity_pct mount
    command -v nvidia-smi >/dev/null 2>&1 && gpu=1
    while _current_metrics; do
        _now_ms now
        lines=""
        if [ $((tick % 2)) -eq 0 ]; then
            if [[ $wants == *" cpu "* ]]; then
                read -r cpu user nice system idle iowait irq softirq steal rest < /proc/stat
                busy=$((user + nice + system + irq + softirq + steal))
                total=$((busy + idle + iowait))
                if [ "$prev_total" -gt 0 ] && [ "$total" -gt "$prev_total" ]; then
                    pct=$((1000 * (busy - prev_busy) / (total - prev_total)))
                    printf -v line '{"type":"metric","name":"cpu","value":%d.%d,"at":%d}' $((pct / 10)) $((pct % 10)) "$now"
                    lines+="$line"$'\n'
                fi
                prev_busy=$busy
                prev_total=$total
            fi
            if [[ $wants == *" mem_used_mb "* || $wants == *" mem_total_mb "* ]]; then
                mem_total=0
                mem_available=0
                while read -r key value unit; do
                    case "$key" in
                        MemTotal:) mem_total=$value ;;
                        MemAvailable:) mem_available=$value ;;
                    esac
                done < /proc/meminfo
                if [ "$mem_total" -gt 0 ]; then
                    if [[ $wants == *" mem_used_mb "* ]]; then
                        printf -v line '{"type":"metric","name":"mem_used_mb","value":%d,"at":%d}' $(((mem_total - mem_available) / 1024)) "$now"
                        lines+="$line"$'\n'
                    fi
                    if [[ $wants == *" mem_total_mb "* ]] && [ $((tick % 60)) -eq 0 ]; then
                        printf -v line '{"type":"metric","name":"mem_total_mb","value":%d,"at":%d}' $((mem_total / 1024)) "$now"
                        lines+="$line"$'\n'
                    fi
                fi
            fi
            if [[ $wants == *" net_rx_kbps "* || $wants == *" net_tx_kbps "* ]]; then
                rx_sum=0
                tx_sum=0
                while IFS=: read -r iface counters; do
                    iface=${iface//[[:space:]]/}
                    if [ -z "$counters" ] || [ "$iface" = lo ]; then
                        continue
                    fi
                    set -- $counters
                    rx_sum=$((rx_sum + $1))
                    tx_sum=$((tx_sum + $9))
                done < /proc/net/dev
                if [ "$prev_net" -gt 0 ] && [ "$now" -gt "$prev_net" ] && [ "$rx_sum" -ge "$prev_rx" ] && [ "$tx_sum" -ge "$prev_tx" ]; then
                    elapsed=$((now - prev_net))
                    if [[ $wants == *" net_rx_kbps "* ]]; then
                        rate=$(((rx_sum - prev_rx) * 80 / elapsed))
                        printf -v line '{"type":"metric","name":"net_rx_kbps","value":%d.%d,"at":%d}' $((rate / 10)) $((rate % 10)) "$now"
                        lines+="$line"$'\n'
                    fi
                    if [[ $wants == *" net_tx_kbps "* ]]; then
                        rate=$(((tx_sum - prev_tx) * 80 / elapsed))
                        printf -v line '{"type":"metric","name":"net_tx_kbps","value":%d.%d,"at":%d}' $((rate / 10)) $((rate % 10)) "$now"
                        lines+="$line"$'\n'
                    fi
                fi
                prev_rx=$rx_sum
                prev_tx=$tx_sum
                prev_net=$now
            fi
        fi
        if [ "$gpu" -eq 1 ] && [ $((tick % 3)) -eq 0 ] && [[ $wants == *" gpu_"* ]]; then
            count=0
            util_sum=0
            used_sum=0
            capacity_sum=0
            temp_max=-1
            power_sum=0
            powered=1
            while IFS=', ' read -r util used capacity temp power; do
                [[ "$util" =~ ^[0-9]+$ && "$used" =~ ^[0-9]+$ && "$capacity" =~ ^[0-9]+$ ]] || continue
                count=$((count + 1))
                util_sum=$((util_sum + util))
                used_sum=$((used_sum + used))
                capacity_sum=$((capacity_sum + capacity))
                if [[ "$temp" =~ ^[0-9]+$ ]] && [ "$temp" -gt "$temp_max" ]; then
                    temp_max=$temp
                fi
                if [[ "$power" =~ ^([0-9]+)(\.([0-9]+))?$ ]]; then
                    cents=${BASH_REMATCH[3]}00
                    power_sum=$((power_sum + 10#${BASH_REMATCH[1]} * 100 + 10#${cents:0:2}))
                else
                    powered=0
                fi
            done < <(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null)
            if [ "$count" -gt 0 ]; then
                if [[ $wants == *" gpu_util "* ]]; then
                    pct=$((10 * util_sum / count))
                    printf -v line '{"type":"metric","name":"gpu_util","value":%d.%d,"at":%d}' $((pct / 10)) $((pct % 10)) "$now"
                    lines+="$line"$'\n'
                fi
                if [[ $wants == *" gpu_mem_mb "* ]]; then
                    printf -v line '{"type":"metric","name":"gpu_mem_mb","value":%d,"at":%d}' "$used_sum" "$now"
                    lines+="$line"$'\n'
                fi
                if [[ $wants == *" gpu_mem_total_mb "* ]] && [ $((tick % 60)) -eq 0 ]; then
                    printf -v line '{"type":"metric","name":"gpu_mem_total_mb","value":%d,"at":%d}' "$capacity_sum" "$now"
                    lines+="$line"$'\n'
                fi
                if [[ $wants == *" gpu_temp_c "* ]] && [ "$temp_max" -ge 0 ]; then
                    printf -v line '{"type":"metric","name":"gpu_temp_c","value":%d,"at":%d}' "$temp_max" "$now"
                    lines+="$line"$'\n'
                fi
                if [[ $wants == *" gpu_power_w "* ]] && [ "$powered" -eq 1 ]; then
                    printf -v line '{"type":"metric","name":"gpu_power_w","value":%d.%02d,"at":%d}' $((power_sum / 100)) $((power_sum % 100)) "$now"
                    lines+="$line"$'\n'
                fi
            fi
        fi
        if [[ $wants == *" disk_used_pct "* ]] && [ $((tick % 30)) -eq 0 ]; then
            while read -r filesystem blocks used_blocks available capacity_pct mount; do
                [[ "$used_blocks" =~ ^[0-9]+$ && "$available" =~ ^[0-9]+$ ]] || continue
                [ $((used_blocks + available)) -gt 0 ] || continue
                pct=$((1000 * used_blocks / (used_blocks + available)))
                printf -v line '{"type":"metric","name":"disk_used_pct","value":%d.%d,"at":%d}' $((pct / 10)) $((pct % 10)) "$now"
                lines+="$line"$'\n'
            done < <(df -Pk / 2>/dev/null)
        fi
        [ -n "$lines" ] && emit "${lines%$'\n'}"
        tick=$((tick + 1))
        sleep 1
    done
}'''
"""The node's own collector: every :data:`~skyward.shared.schemas.Reading`, named in its arguments.

One background loop reads ``/proc/stat``, ``/proc/meminfo`` and ``/proc/net/dev``
with shell builtins and asks ``nvidia-smi`` about every GPU in a single query, so a
tick forks nothing but its ``sleep``, every third second that ``nvidia-smi``, and
every thirtieth a ``df``. What one tick read is appended as one locked write, every
reading stamped with the moment the tick began.

- ``cpu`` (%, every 2s): busy over total jiffies since the previous reading, so the
  first arrives one interval after the loop starts.
- ``mem_used_mb`` (every 2s): ``MemTotal`` minus ``MemAvailable``; ``mem_total_mb``
  every 60s.
- ``net_rx_kbps``, ``net_tx_kbps`` (every 2s): bytes counted on every interface but
  ``lo`` since the previous reading, as kilobits a second — bytes times eight over
  milliseconds is exactly that — so these too arrive one interval late.
- ``gpu_util`` (%, averaged across GPUs), ``gpu_mem_mb`` (summed), ``gpu_temp_c`` (the
  hottest), ``gpu_power_w`` (summed, and only when every GPU reports a draw) every 3s;
  ``gpu_mem_total_mb`` summed every 60s. A machine without ``nvidia-smi``, or one that
  answers with no numeric line, emits none of them.
- ``disk_used_pct`` (every 30s): used over used plus available on ``/``."""

_NOW = """if [ -n "${EPOCHREALTIME:-}" ]; then
    _now_ms() { local micros=${EPOCHREALTIME//[^0-9]/}; printf -v "$1" '%d' $((10#$micros / 1000)); }
else
    _now_ms() { printf -v "$1" '%s' "$(date +%s%3N)"; }
fi"""
"""``_now_ms name`` sets ``name`` to milliseconds since the epoch.

``EPOCHREALTIME`` (bash 5) reads the clock without a fork; its decimal separator
follows the locale, so every non-digit is dropped rather than a ``.``. An older bash
pays a ``date`` per reading."""

_GENERATION = """_metrics_generation="$$.$RANDOM"
printf '%s\\n' "$_metrics_generation" > /opt/skyward/metrics.generation
_current_metrics() { local current; { read -r current < /opt/skyward/metrics.generation; } 2>/dev/null; [ "$current" = "$_metrics_generation" ]; }"""
"""Which bootstrap's collectors are the live ones.

Each run writes a fresh token and every collector loop checks it before each sample,
so a bootstrap re-run on the same machine makes the previous run's loops exit within
one interval — nothing is killed, so no PID can be reused into the wrong process."""


def _collector(name: str, command: str, interval: float) -> str:
    """One custom metric as a background shell loop: read, emit if numeric, sleep, repeat.

    ``set +e`` because a sample is allowed to fail — a busy ``nvidia-smi``, a missing
    interface — without taking the loop down with it; the next tick tries again. The
    regex is the gate: only a bare number reaches ``emit_metric``, so a command that
    printed a warning where a value should have been is dropped, not written as one.
    The loop ends once a newer bootstrap has taken over the generation.
    """
    return "\n".join(
        (
            f"_collect_{name}() {{",
            "    set +e",
            "    local v now",
            "    while _current_metrics; do",
            f"        v=$({command})",
            "        _now_ms now",
            f'        [[ "$v" =~ ^-?[0-9]*\\.?[0-9]+$ ]] && emit_metric {name} "$v" "$now"',
            f"        sleep {interval}",
            "    done",
            "}",
        ),
    )


def metrics(specs: Sequence[Reading | MetricSpec] | None = None) -> str:
    """The collectors, and the one call that sets them going.

    Started before the bootstrap phases and left running as background jobs: the
    script is ``nohup``-ed and non-interactive, so the loops outlive it and keep
    reporting while the worker runs — and still report if the worker never does.

    The readings named go to the one built-in loop, each :class:`MetricSpec` to a loop
    of its own; ``None`` names every reading. Either way the output first claims the
    generation, so the collectors of an earlier run on the same machine stop.
    """
    readings: list[Reading] = []
    commands: list[MetricSpec] = []
    for metric in READINGS if specs is None else specs:
        match metric:
            case MetricSpec():
                commands.append(metric)
            case reading:
                readings.append(reading)

    emit_metric = 'emit_metric() { emit "{\\"type\\":\\"metric\\",\\"name\\":\\"$1\\",\\"value\\":$2,\\"at\\":$3}"; }'
    builtin = ((_BUILTIN, f"    _collect_builtin {' '.join(readings)} &"),) if readings else ()
    loops = ((_collector(spec.name, spec.command, spec.interval), f"    _collect_{spec.name} &") for spec in commands)
    collectors, starts = zip(*builtin, *loops, strict=True) if readings or commands else ((), ())
    return "\n".join(
        (
            _GENERATION,
            _NOW,
            emit_metric,
            *collectors,
            "start_metrics_daemon() {",
            *starts,
            "    :",
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


def mounts(volumes: tuple[tuple[Volume, Endpoint], ...]) -> str:
    """The ``volumes`` phase: install geesefs, mount every bucket, link it where it was asked for.

    One FUSE mount per ``(bucket, endpoint)`` pair rather than per volume, because
    two volumes reading different prefixes of one bucket are one filesystem seen
    twice, and mounting it twice would double the cache and the connections for
    nothing. The prefixes then become symlinks into the single mount, which is also
    what makes ``prefix`` cost nothing.

    A pair is mounted read-write if any volume naming it asked to write, so a bucket
    that is read-only in one place and written in another is writable — the narrower
    of the two would fail the write, and the mount is shared.

    Credentials are written per endpoint, once, as an AWS shared-config file at mode
    600. An endpoint with no access key is signed for by the machine's own instance
    identity instead, which is why an AWS bucket in the account that bought the
    machine needs no secret to travel anywhere.

    Parameters
    ----------
    volumes : tuple[tuple[Volume, Endpoint], ...]
        Each volume paired with the endpoint it is reached through.

    Returns
    -------
    str
        A single ``phase`` line. Its commands are chained on success, so the first
        that fails is what the journal reports.
    """
    lock = "-o DPkg::Lock::Timeout=-1"
    commands = [
        f"apt-get {lock} install -y -qq ca-certificates curl",
        f"{{ apt-get {lock} install -y -qq fuse3 || apt-get {lock} install -y -qq fuse; }}",
        "arch=$(uname -m)",
        'case "$arch" in x86_64) a=amd64 ;; aarch64|arm64) a=arm64 ;; *) echo "geesefs: unsupported arch $arch" >&2; exit 1 ;; esac',
        f'curl -fsSL -o {GEESEFS_BIN} "https://github.com/yandex-cloud/geesefs/releases/download/{GEESEFS}/geesefs-linux-${{a}}"',
        f"chmod +x {GEESEFS_BIN}",
    ]

    endpoints = {endpoint.url: endpoint for _, endpoint in volumes}
    credentials: dict[str, str] = {}
    for url, endpoint in endpoints.items():
        if endpoint.access_key is None:
            continue
        path = f"/etc/geesefs-creds-{hashlib.md5(url.encode()).hexdigest()[:8]}"
        block = f"[default]\naws_access_key_id = {endpoint.access_key}\naws_secret_access_key = {endpoint.secret_key}\n"
        commands.append(f"printf %s {shlex.quote(block)} > {path}")
        commands.append(f"chmod 600 {path}")
        credentials[url] = path

    writable: dict[tuple[str, str], bool] = {}
    for volume, endpoint in volumes:
        target = (volume.bucket, endpoint.url)
        writable[target] = writable.get(target, False) or not volume.read_only

    for (bucket, url), rewritable in writable.items():
        endpoint = endpoints[url]
        mount = f"{FUSE_ROOT}/{bucket}"
        log = f"/tmp/geesefs-{bucket}.log"
        flags = [f"--endpoint={url}"]
        if not endpoint.path_style:
            flags.append("--subdomain")
        flags.append(f"--shared-config={credentials[url]}" if url in credentials else "--iam --iam-flavor=imdsv1")
        flags += ["--stat-cache-ttl=1s", f"--log-file={log}"]
        options = "allow_other" if rewritable else "allow_other,ro"
        commands.append(f"mkdir -p {mount}")
        commands.append(f"{GEESEFS_BIN} {' '.join(flags)} -o {options} {bucket} {mount}")
        commands.append(f"mountpoint -q {mount} || {{ cat {log} 2>/dev/null; echo 'geesefs: failed to mount {mount}' >&2; exit 1; }}")

    for volume, _ in volumes:
        source = f"{FUSE_ROOT}/{volume.bucket}/{volume.prefix}" if volume.prefix else f"{FUSE_ROOT}/{volume.bucket}"
        if volume.prefix and not volume.read_only:
            commands.append(f"mkdir -p {source}")
        commands.append(f"ln -sfn {source} {volume.mount}")

    return phase("volumes", *commands)


def symlinks(volumes: tuple[Volume, ...], base: str) -> str:
    """The ``volumes`` phase for a machine whose volume the host already mounted.

    No install and no FUSE: the provider attached the storage at ``base`` before the
    machine booted, so all that is owed is the path the user asked for. Each prefix
    becomes a subdirectory of ``base``; a volume with none links ``base`` itself.
    """
    commands: list[str] = []
    for volume in volumes:
        target = f"{base}/{volume.prefix}".rstrip("/") if volume.prefix else base
        commands.append(f"mkdir -p {target}")
        commands.append(f"ln -sfn {target} {volume.mount}")
    return phase("volumes", *commands)


def _package_name(spec: str) -> str:
    """The bare name a requirement scopes to, dropping any version or extras."""
    return re.split(r"[<>=!~ \[]", spec, maxsplit=1)[0]


def _apt(extra: Sequence[str]) -> str:
    """Install ``curl`` and ``git`` — the uv installer and git sources need them — plus extras.

    Needed because a minimal base image (RunPod) ships without ``curl``, so the
    fallback that installs uv is ``curl: command not found`` and the machine never
    gets a venv.

    Guarded because most images ship with both, and running it anyway is an
    ``apt-get update`` against the distribution's mirrors on every node of every
    pool — measured at 25.6s of a 26s bootstrap on a warm image, against 0.2s for
    uv, the venv and installing skyward put together. An image the machine
    already has is exactly the case that pays it for nothing.

    The guard is on the two binaries because they are binaries. An ``apt`` the
    user asked for may be a library with no command to look for, so a spec that
    names extras installs unconditionally: they were asked for, and only the
    caller knows what they are.
    """
    packages = " ".join(("curl", "git", *extra))
    install = f"apt-get update -qq && apt-get install -y -qq {packages}"
    if extra:
        return f"phase apt '{install}'"
    return f"phase apt 'command -v curl >/dev/null 2>&1 && command -v git >/dev/null 2>&1 || {{ {install}; }}'"


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


def script(image: Image, skyward: str, plugins: tuple[Plugin, ...] = (), concurrency: int = 1, volumes: tuple[str, ...] = ()) -> str:
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
    volumes : tuple[str, ...]
        Phases that mount the compute's buckets, rendered by the daemon at bind
        time because that is where the credentials are. They run before the
        plugins', so a plugin that caches into a volume finds it there.

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
    exports = "\n".join(f"export {key}={value!r}" for key, value in image.env.items())
    env_file = f"cat > {ENV} <<'ENVSH'\n{exports}\nENVSH" if exports else ""
    preload = f"source {VARS} 2>/dev/null; " if image.shell_vars else ""

    if image.pip_indexes:
        pyproject = f"cat > {SKYWARD_DIR}/pyproject.toml <<'PYPROJECT'\n{_pyproject(image)}\nPYPROJECT"
        install = f"cd {SKYWARD_DIR} && uv add --python {PYTHON}"
    else:
        pyproject = ""
        install = f"uv pip install --python {PYTHON}"

    postamble = "\n".join((*volumes, *(op for plugin in plugins for op in plugin.bootstrap(image, concurrency))))

    return "\n".join(
        (
            HEADER,
            metrics(image.metrics),
            _shell_vars(image.shell_vars) if image.shell_vars else "",
            env_file,
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


RESTARTS = 5
"""How many deaths in a row, each within ``SHORT_LIVED`` seconds of starting, before the machine is given up on."""

SHORT_LIVED = 30
"""Seconds a worker has to stay up for its death to count as a crash rather than as a crash loop."""


def supervised(command: str) -> str:
    """``command``, started again whenever it dies, as a script for ``bash -c``.

    On the thread executor the user's function runs inside the worker's own process,
    and a function that crashes the interpreter — a segfault in an extension, bytecode
    compiled by another Python — takes the worker with it. The machine is fine and
    nothing on it answers, and the daemon would only learn so the next time the link
    dropped. Started again, the worker answers the daemon's next question about the
    attempt it was running by not knowing it, which is already how an attempt is
    declared lost.

    An exit of zero is the worker leaving on purpose — a failed health check writes
    its reason to the journal and returns — so it is not started again. A worker that
    dies over and over just after starting is broken rather than unlucky: after
    ``RESTARTS`` of those in a row the loop stops and says so as a health event, which
    the daemon already reads as the node being lost.
    """
    return f"""\
[ -f {ENV} ] && . {ENV}
. {SKYWARD_DIR}/emit.sh
set +e
quick=0
while true; do
    began=$(date +%s)
    {command}
    code=$?
    [ "$code" -eq 0 ] && exit 0
    if [ $(( $(date +%s) - began )) -lt {SHORT_LIVED} ]; then quick=$((quick + 1)); else quick=0; fi
    if [ "$quick" -ge {RESTARTS} ]; then
        emit "{{\\"type\\":\\"health\\",\\"reason\\":\\"the worker died $quick times in a row within {SHORT_LIVED}s of starting, last with code $code\\"}}"
        exit "$code"
    fi
    emit_console "the worker exited with code $code; starting it again"
    sleep 1
done
"""
