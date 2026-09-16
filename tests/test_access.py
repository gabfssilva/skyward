"""Reaching the machines for something other than running a function.

A pool is a handful of computers the user is paying for, so the questions asked
here are the ones asked of a computer: what is on it, put this there, and let me
talk to the port that thing is listening on.
"""

import asyncio
import os
import pty
import subprocess
import sys
import time
import urllib.request
from contextlib import suppress
from pathlib import Path

import cloudpickle
import pytest

import skyward as sky
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

        ran = cli("compute", "run", pool.id, str(script), "--all", "--url", daemon)

        assert ran.code == 0, ran.err
        assert "spoke from rank 0" in ran.out, "the lines come back over the event log, and this is where they land"
        assert "spoke from rank 1" in ran.out


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
