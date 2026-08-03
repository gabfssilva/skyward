"""Reaching the machines for something other than running a function.

A pool is a handful of computers the user is paying for, so the questions asked
here are the ones asked of a computer: what is on it, put this there, and let me
talk to the port that thing is listening on.
"""

import sys
import urllib.request
from pathlib import Path

import cloudpickle
import pytest

import skyward as sky
from tests.conftest import Build, cli, rows

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


def describe_a_port_on_a_node() -> None:
    def it_is_reachable_on_loopback_for_as_long_as_the_block_lasts(compute: Build) -> None:
        with compute(ports=[sky.Port(remote=PORT, local=PORT)]) as pool:
            node = serve(PORT) >> pool

            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}", timeout=30) as answer:
                assert answer.read().decode() == node, "the loopback port reached the node that answered"
