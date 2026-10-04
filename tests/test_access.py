"""Reaching the machines for something other than running a function.

A pool is a handful of computers the user is paying for, so the questions asked
here are the ones asked of a computer: what is on it, put this there, and let me
talk to the port that thing is listening on.
"""

import asyncio
import os
import pty
import signal
import subprocess
import sys
import time
import urllib.request
from contextlib import suppress
from pathlib import Path

import cloudpickle
import httpx
import pytest

import skyward as sky
from skyward.shared import codec
from tests.conftest import SKY, Build, cli, rows

pytest.importorskip("cyclopts", reason="the sky CLI needs: pip install 'skyward[cli]'")

pytestmark = [pytest.mark.compute, pytest.mark.xdist_group("pool")]

cloudpickle.register_pickle_by_value(sys.modules[__name__])

PORT = 18_231


@sky.function
def serve(port: int) -> str:
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Hello(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self.send_response(200)
            self.end_headers()
            self.wfile.write(sky.instance_info().node.encode())

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

    server = HTTPServer(("0.0.0.0", port), Hello)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    return sky.instance_info().node


def describe_running_a_command_on_the_machines() -> None:
    def it_answers_for_every_node_at_once(pool: sky.Compute, daemon: str) -> None:
        ran = rows("compute", "exec", pool.id, "echo", "alive", "--url", daemon)

        assert len(ran) == 2, "one answer per machine"
        assert all("alive" in str(row) for row in ran)


    def it_answers_the_same_asked_by_name(pool: sky.Compute, daemon: str) -> None:
        by_name = rows("compute", "exec", "shared", "echo", "alive", "--url", daemon)
        by_id = rows("compute", "exec", pool.id, "echo", "alive", "--url", daemon)

        assert by_name == by_id, "a name and an id name the same compute everywhere else"


def describe_running_a_script_on_the_machines() -> None:
    def it_prints_what_the_script_printed(pool: sky.Compute, daemon: str, tmp_path: Path) -> None:
        script = tmp_path / "speak.py"
        script.write_text("import skyward as sky\nprint(f'spoke from rank {sky.instance_info().rank}')\n")

        ran = cli("compute", "run", pool.id, str(script), "--node", "all", "--url", daemon)

        assert ran.code == 0, ran.err
        assert "0 │ spoke from rank 0" in ran.out, "the lines come back over the event log, after the rank that wrote them"
        assert "1 │ spoke from rank 1" in ran.out

    def a_rank_runs_it_on_that_node_alone(pool: sky.Compute, daemon: str, tmp_path: Path) -> None:
        script = tmp_path / "rank.py"
        script.write_text("import skyward as sky\nprint(f'spoke from rank {sky.instance_info().rank}')\n")

        ran = cli("compute", "run", pool.id, str(script), "--node", "1", "--url", daemon)

        assert ran.code == 0, ran.err
        assert ran.out.splitlines() == ["1 │ spoke from rank 1"]

    def without_a_node_it_runs_once(pool: sky.Compute, daemon: str, tmp_path: Path) -> None:
        script = tmp_path / "once.py"
        script.write_text("print('once')\n")

        ran = cli("compute", "run", pool.id, str(script), "--url", daemon)

        assert ran.code == 0, ran.err
        assert [line.split(" │ ", 1)[1] for line in ran.out.splitlines()] == ["once"]

    def it_refuses_a_placement_that_is_not_one(pool: sky.Compute, daemon: str, tmp_path: Path) -> None:
        script = tmp_path / "nowhere.py"
        script.write_text("print('never')\n")

        ran = cli("compute", "run", pool.id, str(script), "--node", "nowhere", "--url", daemon)

        assert ran.code != 0
        assert "--node takes all, any or a rank" in ran.err

    def it_keeps_the_script_as_the_text_of_its_function(pool: sky.Compute, daemon: str, tmp_path: Path) -> None:
        """The script travels as a function of skyward's own bound to its text, so the text is the only part of it worth reading."""
        script = tmp_path / "kept.py"
        script.write_text("import sys\n\nprint(sys.argv)\n")

        ran = cli("compute", "run", pool.id, str(script), "--url", daemon)
        assert ran.code == 0, ran.err

        listed = httpx.get(f"{daemon}/v1/functions", params={"latest": "true", "limit": 500}, timeout=30)
        assert listed.status_code == 200, listed.text
        assert [function["excerpt"] for function in listed.json()["items"] if function["name"] == "kept.py"] == [script.read_text()]


def describe_a_terminal_on_a_machine() -> None:
    def it_carries_what_is_typed_and_what_is_painted(pool: sky.Compute, daemon: str) -> None:
        """``\x04`` is the end of input a terminal has: a pipe closing is not one, since a tty never closes."""
        ran = cli("compute", "ssh", pool.id, "--node", "0", "--command", "cat", "--url", daemon, stdin="typed at the machine\n\x04")

        assert ran.code == 0, ran.err
        assert "typed at the machine" in ran.out, "up carried the keystrokes, down carried what the pty painted"


    def it_paints_a_burst_bigger_than_the_terminal_holds(pool: sky.Compute, daemon: str) -> None:
        """Watching the keyboard through the loop puts the terminal in non-blocking mode, and stdout is that same terminal."""
        code, painted = _under_a_tty("compute", "ssh", pool.id, "--node", "0", "--command", "head -c 200000 /dev/zero | tr '\\0' x", "--url", daemon)

        assert "BlockingIOError" not in painted, "a plain write to a terminal it filled is refused, and the session dies of it"
        assert code == 0
        assert painted.count("x") > 100_000, "and every byte still arrives, held until the terminal takes it"

    def it_is_reachable_over_one_socket(pool: sky.Compute, daemon: str) -> None:
        """What the browser console uses, and which uvicorn serves only with a WebSocket library installed."""
        painted = asyncio.run(_over_a_socket(daemon, pool.id))

        assert "typed over a socket" in painted, "keystrokes went up the same socket the terminal painted down"

    def a_machine_nobody_holds_closes_the_socket_with_the_reason(pool: sky.Compute, daemon: str) -> None:
        """A browser is told nothing by a rejected upgrade, so the refusal is a frame and the close carries its code."""
        code, refusal = asyncio.run(_refused_over_a_socket(daemon, pool.id))

        assert "compute_not_connected" in refusal, "the same error every other endpoint answers with"
        assert code == 4409, "and the close says so too, for a caller that only listens for one"


def describe_a_function_written_rather_than_pickled() -> None:
    def it_runs_on_the_machine_it_was_pointed_at(pool: sky.Compute, daemon: str) -> None:
        """What the browser console does, end to end, with no interpreter anywhere on the caller's side."""
        source = "import skyward as sky\n\n\ndef greet(name):\n    return f'{name} from rank {sky.instance_info().rank}'\n"

        written = httpx.post(f"{daemon}/v1/functions", json={"name": "greet", "source": source}, timeout=30)
        assert written.status_code in (200, 201), written.text

        for rank in (1, 0):
            said, recorded = _greeted(daemon, pool.id, written.json()["sha256"], rank)

            assert said == f"typed in a browser from rank {rank}", "the source ran where it was pointed, on the argument it was given"
            assert recorded == rank, "and the attempt is written down under the machine that took it"


def _greeted(daemon: str, compute: str, function: str, rank: int) -> tuple[str, int]:
    """One task on one named machine, and what it said. The arguments never become Python until the daemon has them."""
    submitted = httpx.post(
        f"{daemon}/v1/tasks",
        json={
            "compute": compute,
            "function": function,
            "dispatch": "one",
            "rank": rank,
            "call": {"args": ["typed in a browser"]},
        },
        headers={"Idempotency-Key": os.urandom(16).hex()},
        timeout=30,
    )
    assert submitted.status_code == 201, submitted.text

    task = submitted.json()["id"]
    said: str = codec.loads(_settled(daemon, task))
    [attempt] = httpx.get(f"{daemon}/v1/tasks/{task}", timeout=30).json()["executions"]
    return said, attempt["rank"]


def _settled(daemon: str, task: str, timeout: float = 180.0) -> bytes:
    """The result, once there is one. A 204 is the daemon saying to ask again."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        answer = httpx.get(f"{daemon}/v1/tasks/{task}/result", params={"wait": 10}, timeout=30)
        if answer.status_code == 200:
            return answer.content
        assert answer.status_code == 204, answer.text
    raise AssertionError(f"task {task} never settled")


def describe_putting_a_file_on_the_machines() -> None:
    def it_lands_on_every_node_and_comes_back_off_one(pool: sky.Compute, daemon: str, tmp_path: Path) -> None:
        source = tmp_path / "payload.txt"
        source.write_text("carried by hand\n")

        assert cli("compute", "upload", pool.id, str(source), "/tmp/payload.txt", "--url", daemon).code == 0

        listed = rows("compute", "ls", pool.id, "/tmp/payload.txt", "--url", daemon, "--node", "all")
        assert len(listed) == 2, "both machines have it"

        back = tmp_path / "back.txt"
        assert cli("compute", "download", pool.id, "/tmp/payload.txt", str(back), "--url", daemon).code == 0

        assert back.read_text() == "carried by hand\n"


def describe_mirroring_a_folder_off_a_machine() -> None:
    def it_brings_the_tree_down_and_then_only_what_changed(pool: sky.Compute, daemon: str, tmp_path: Path) -> None:
        _on(daemon, pool.id, "0", "rm -rf /tmp/mirrored && mkdir -p /tmp/mirrored/deep")
        _on(daemon, pool.id, "0", "echo one > /tmp/mirrored/a.txt && echo two > /tmp/mirrored/deep/b.txt")
        local = tmp_path / "mirror"

        first = cli("compute", "sync", pool.id, "/tmp/mirrored", str(local), "--url", daemon)

        assert first.code == 0, first.err
        assert (local / "a.txt").read_text() == "one\n"
        assert (local / "deep" / "b.txt").read_text() == "two\n"
        assert "2 files" in first.out

        _on(daemon, pool.id, "0", "echo three > /tmp/mirrored/deep/b.txt")
        second = cli("compute", "sync", pool.id, "/tmp/mirrored", str(local), "--url", daemon)

        assert "1 files" in second.out, "the file nobody touched stays where it is"
        assert (local / "deep" / "b.txt").read_text() == "three\n"

    def every_node_lands_in_a_folder_of_its_rank(pool: sky.Compute, daemon: str, tmp_path: Path) -> None:
        _on(daemon, pool.id, "all", "rm -rf /tmp/ranked && mkdir -p /tmp/ranked && hostname > /tmp/ranked/who.txt")
        local = tmp_path / "ranked"

        assert cli("compute", "sync", pool.id, "/tmp/ranked", str(local), "--node", "all", "--url", daemon).code == 0

        assert sorted(path.name for path in local.iterdir()) == ["0", "1"]
        assert (local / "0" / "who.txt").read_text() != (local / "1" / "who.txt").read_text()

    def a_folder_that_is_not_there_is_said_so(pool: sky.Compute, daemon: str, tmp_path: Path) -> None:
        ran = cli("compute", "sync", pool.id, "/tmp/never-made", str(tmp_path / "nothing"), "--url", daemon)

        assert ran.code != 0
        assert "/tmp/never-made" in ran.err

    def watching_brings_down_what_is_written_afterwards(pool: sky.Compute, daemon: str, tmp_path: Path) -> None:
        _on(daemon, pool.id, "0", "rm -rf /tmp/watched && mkdir -p /tmp/watched")
        local = tmp_path / "watched"
        watching = subprocess.Popen(
            [str(SKY), "compute", "sync", pool.id, "/tmp/watched", str(local), "--watch", "--interval", "1", "--url", daemon],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            _on(daemon, pool.id, "0", "echo late > /tmp/watched/late.txt")
            deadline = time.monotonic() + 60
            while not (local / "late.txt").exists() and time.monotonic() < deadline:
                time.sleep(0.2)

            assert (local / "late.txt").read_text() == "late\n"
            assert watching.poll() is None, "one pass is not the end of a watch"
        finally:
            watching.terminate()
            watching.wait(timeout=30)


    def an_interrupted_watch_makes_one_last_pass(pool: sky.Compute, daemon: str, tmp_path: Path) -> None:
        _on(daemon, pool.id, "0", "rm -rf /tmp/parting && mkdir -p /tmp/parting && echo first > /tmp/parting/first.txt")
        local = tmp_path / "parting"
        watching = subprocess.Popen(
            [str(SKY), "compute", "sync", pool.id, "/tmp/parting", str(local), "--watch", "--interval", "600", "--url", daemon],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            deadline = time.monotonic() + 60
            while not (local / "first.txt").exists() and time.monotonic() < deadline:
                time.sleep(0.2)
            _on(daemon, pool.id, "0", "echo last > /tmp/parting/last.txt")

            watching.send_signal(signal.SIGINT)
            watching.wait(timeout=60)
        finally:
            watching.kill()

        assert (local / "last.txt").read_text() == "last\n", "written inside the interval, and still brought down"

    def a_watch_ends_with_the_compute_it_was_watching(compute: Build, daemon: str, tmp_path: Path) -> None:
        with compute() as pool:
            _on(daemon, pool.id, "0", "mkdir -p /tmp/short-lived")
            watching = subprocess.Popen(
                [str(SKY), "compute", "sync", pool.id, "/tmp/short-lived", str(tmp_path / "gone"), "--watch", "--interval", "1", "--url", daemon],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        try:
            _, said = watching.communicate(timeout=60)
        finally:
            watching.kill()

        assert watching.returncode != 0
        assert pool.id in said


def _on(daemon: str, compute: str, node: str, command: str) -> None:
    ran = cli("compute", "exec", compute, command, "--node", node, "--url", daemon)
    assert ran.code == 0, ran.err


async def _over_a_socket(daemon: str, compute: str) -> str:
    """Attach to a machine's terminal the way the console does, and return what it painted."""
    from websockets.asyncio.client import connect

    url = f"{daemon.replace('http://', 'ws://')}/v1/computes/{compute}/shell/attach?node=0&command=cat"
    async with connect(url) as socket:
        await socket.send(b"typed over a socket\n")
        painted = ""
        while "typed over a socket" not in painted:
            frame = await asyncio.wait_for(socket.recv(), timeout=60)
            painted += frame.decode(errors="replace") if isinstance(frame, bytes) else frame
        return painted


async def _refused_over_a_socket(daemon: str, compute: str) -> tuple[int, str]:
    """Ask for a rank the daemon holds no machine at, and return how the socket ended."""
    from websockets.asyncio.client import connect
    from websockets.exceptions import ConnectionClosed

    url = f"{daemon.replace('http://', 'ws://')}/v1/computes/{compute}/shell/attach?node=97"
    async with connect(url) as socket:
        refusal = await asyncio.wait_for(socket.recv(), timeout=60)
        with suppress(ConnectionClosed):
            await asyncio.wait_for(socket.recv(), timeout=60)
        return socket.close_code or 0, str(refusal)


def _under_a_tty(*tokens: str) -> tuple[int, str]:
    """Run ``sky`` with a terminal on all three of its streams, and read what it painted.

    Nothing is read for a moment on purpose: the terminal's own buffer is small, and
    the case under test only exists once the far end has filled it and the writer has
    to wait.
    """
    master, slave = pty.openpty()
    with subprocess.Popen([str(SKY), *tokens], stdin=slave, stdout=slave, stderr=slave) as running:
        os.close(slave)
        time.sleep(1.0)
        painted = bytearray()
        with suppress(OSError):
            while data := os.read(master, 65536):
                painted += data
        os.close(master)
        return running.wait(timeout=180), painted.decode(errors="replace")


def describe_a_port_on_a_node() -> None:
    def it_is_reachable_on_loopback_for_as_long_as_the_block_lasts(compute: Build) -> None:
        with compute(ports=[sky.Port(remote=PORT, local=PORT)]) as pool:
            node = serve(PORT) >> pool

            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}", timeout=30) as answer:
                assert answer.read().decode() == node, "the loopback port reached the node that answered"
