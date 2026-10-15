"""The metrics collectors a bootstrap starts, as the shell that runs them sees them."""

import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import get_args

import msgspec
import pytest

import skyward as sky
from skyward.shared.schemas import READINGS, Endpoint, Image, MetricSpec, Reading, Volume
from skyward.worker import bootstrap

IMAGE = "ghcr.io/gabfssilva/skyward:py3.13"

RUNNER = """
cat > /tmp/bootstrap.sh
bash /tmp/bootstrap.sh
sleep 5
grep '"type":"metric"' /opt/skyward/events.jsonl > /tmp/before
cat /tmp/before
echo ---
printf 'newer\\n' > /opt/skyward/metrics.generation
sleep 3
first=$(grep -c '"type":"metric"' /opt/skyward/events.jsonl)
sleep 4
second=$(grep -c '"type":"metric"' /opt/skyward/events.jsonl)
echo "$first $second"
"""


def _syntax(tmp_path: Path, text: str) -> subprocess.CompletedProcess[str]:
    path = tmp_path / "metrics.sh"
    path.write_text(bootstrap.HEADER + "\n" + text)
    return subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True, timeout=10, check=False)


def describe_the_metrics_script() -> None:
    @pytest.mark.local
    def it_is_valid_bash_by_default(tmp_path: Path) -> None:
        done = _syntax(tmp_path, bootstrap.metrics(None))

        assert done.returncode == 0, done.stderr

    @pytest.mark.local
    def it_is_valid_bash_with_readings_and_custom_specs(tmp_path: Path) -> None:
        specs = (
            "cpu",
            "gpu_power_w",
            MetricSpec(name="load", command="cut -d' ' -f1 /proc/loadavg", interval=1),
            MetricSpec(name="disk", command="df / | awk 'NR==2 {print $5}' | tr -d %", interval=2.5),
        )

        done = _syntax(tmp_path, bootstrap.metrics(specs))

        assert done.returncode == 0, done.stderr

    @pytest.mark.local
    def it_is_valid_bash_when_nothing_is_measured(tmp_path: Path) -> None:
        done = _syntax(tmp_path, bootstrap.metrics(()))

        assert done.returncode == 0, done.stderr

    @pytest.mark.local
    def no_metrics_named_is_every_reading_the_default_names() -> None:
        assert bootstrap.metrics(None) == bootstrap.metrics(sky.metrics.Default())


def describe_the_readings() -> None:
    @pytest.mark.local
    def the_default_lists_every_one_of_them() -> None:
        assert sky.metrics.Default() == READINGS
        assert set(READINGS) == set(get_args(Reading.__value__))

    @pytest.mark.local
    def an_image_keeps_the_defaults_beside_its_own() -> None:
        loss = sky.metrics.Custom("loss", "cat /tmp/loss")

        image = Image(metrics=[*sky.metrics.Default(), loss])

        assert msgspec.json.decode(msgspec.json.encode(image), type=Image) == image

    @pytest.mark.local
    def a_name_given_twice_is_refused() -> None:
        with pytest.raises(ValueError, match="cpu"):
            Image(metrics=["cpu", MetricSpec(name="cpu", command="echo 1", interval=1)])


def _collected(script: str, prelude: str = "") -> tuple[list[dict[str, object]], str]:
    done = subprocess.run(
        ["docker", "run", "--rm", "-i", IMAGE, "bash", "-c", prelude + RUNNER],
        input=bootstrap.HEADER + "\n" + script + "\n",
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    metrics, counts = done.stdout.split("---\n")
    return [json.loads(line) for line in metrics.splitlines()], counts


def _nvidia_smi(*lines: str) -> str:
    """A stand-in ``nvidia-smi`` that answers every query with these lines, one per GPU."""
    answer = "\n".join(lines)
    return f"cat > /usr/local/bin/nvidia-smi <<'SMI'\n#!/bin/sh\ncat <<'EOF'\n{answer}\nEOF\nSMI\nchmod +x /usr/local/bin/nvidia-smi\n"


def describe_the_builtin_loop_on_linux() -> None:
    @pytest.mark.compute
    @pytest.mark.xdist_group("bootstrap")
    def it_reports_every_reading_this_machine_has_and_stops_when_a_newer_generation_takes_over() -> None:
        started = time.time_ns() // 1_000_000

        records, counts = _collected(bootstrap.metrics(None))

        names = {record["name"] for record in records}
        assert {"cpu", "mem_used_mb", "mem_total_mb", "net_rx_kbps", "net_tx_kbps", "disk_used_pct"} <= names <= set(READINGS)
        assert all(set(record) == {"type", "name", "value", "at"} and isinstance(record["value"], int | float) for record in records)
        assert all(isinstance(record["at"], int) and started - 60_000 < record["at"] < started + 60_000 for record in records)
        first, second = counts.split()
        assert first == second

    @pytest.mark.compute
    @pytest.mark.xdist_group("bootstrap")
    def it_reports_only_the_readings_named_beside_the_commands() -> None:
        script = bootstrap.metrics(("cpu", "disk_used_pct", MetricSpec(name="load", command="cut -d' ' -f1 /proc/loadavg", interval=1)))

        records, _ = _collected(script)

        assert {record["name"] for record in records} == {"cpu", "disk_used_pct", "load"}
        assert all(isinstance(record["at"], int) for record in records)

    @pytest.mark.compute
    @pytest.mark.xdist_group("bootstrap")
    def it_reads_every_gpu_in_one_query_averaging_summing_and_taking_the_hottest() -> None:
        records, _ = _collected(bootstrap.metrics(None), _nvidia_smi("80, 1000, 24000, 65, 250.50", "90, 3000, 24000, 71, 300.25"))

        readings = {record["name"]: record["value"] for record in records if str(record["name"]).startswith("gpu_")}
        assert readings == {"gpu_util": 85.0, "gpu_mem_mb": 4000, "gpu_mem_total_mb": 48000, "gpu_temp_c": 71, "gpu_power_w": 550.75}

    @pytest.mark.compute
    @pytest.mark.xdist_group("bootstrap")
    def a_gpu_that_does_not_report_a_reading_leaves_that_reading_out() -> None:
        records, _ = _collected(bootstrap.metrics(None), _nvidia_smi("80, 1000, 24000, 65, [N/A]", "90, 3000, 24000, [N/A], 300.25"))

        names = {record["name"] for record in records if str(record["name"]).startswith("gpu_")}
        assert names == {"gpu_util", "gpu_mem_mb", "gpu_mem_total_mb", "gpu_temp_c"}


def describe_the_worker_supervisor() -> None:
    @pytest.mark.local
    def it_is_valid_bash(tmp_path: Path) -> None:
        path = tmp_path / "supervised.sh"
        path.write_text(bootstrap.supervised("true"))

        assert subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True, timeout=10, check=False).returncode == 0

    @pytest.mark.compute
    @pytest.mark.xdist_group("bootstrap")
    def it_starts_a_worker_that_crashed_again_and_lets_one_that_left_go() -> None:
        """Two segfaults, then a clean exit: the way a failed health check leaves."""
        events, code = _supervised('n=$(cat /tmp/runs 2>/dev/null || echo 0)\necho $((n + 1)) > /tmp/runs\n[ "$n" -ge 2 ] && exit 0\nkill -SEGV $$')

        assert code == 0
        assert [event["content"] for event in events if event["type"] == "console"] == ["the worker exited with code 139; starting it again"] * 2
        assert not [event for event in events if event["type"] == "health"]

    @pytest.mark.compute
    @pytest.mark.xdist_group("bootstrap")
    def it_gives_the_machine_up_when_the_worker_dies_every_time_it_starts() -> None:
        events, code = _supervised("exit 3")

        assert code == 3
        assert events[-1] == {
            "type": "health",
            "reason": f"the worker died {bootstrap.RESTARTS} times in a row within {bootstrap.SHORT_LIVED}s of starting, last with code 3",
        }


def _supervised(worker: str) -> tuple[list[dict[str, object]], int]:
    """The supervisor over a stand-in worker, on a machine that has been bootstrapped: its journal, and how it ended."""
    runner = "\n".join(
        (
            f"cat > /tmp/worker.sh <<'WORKER'\n{worker}\nWORKER",
            f"cat > /tmp/bootstrap.sh <<'BOOTSTRAP'\n{bootstrap.HEADER}{bootstrap.FOOTER}BOOTSTRAP",
            "bash /tmp/bootstrap.sh",
            f"bash -c {shlex.quote(bootstrap.supervised('bash /tmp/worker.sh'))}",
            'echo "exit $?"',
            "grep -v '\"type\":\"phase\"' /opt/skyward/events.jsonl",
        )
    )
    done = subprocess.run(["docker", "run", "--rm", "-i", IMAGE, "bash"], input=runner, capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 0, done.stderr
    ended, *lines = done.stdout.splitlines()
    return [json.loads(line) for line in lines], int(ended.removeprefix("exit "))


def describe_the_refresh_script() -> None:
    @pytest.mark.local
    def it_installs_the_packages_and_leaves_the_journal_and_the_venv_alone() -> None:
        script = bootstrap.refresh(Image(pip=["six"]))

        assert "phase deps" in script
        assert "emit_phase started refresh" in script
        assert "emit_phase completed refresh" in script
        assert "uv venv" not in script
        assert "rm -f /opt/skyward/events.jsonl" not in script
        assert "emit_phase started bootstrap" not in script

    @pytest.mark.local
    def a_bootstrap_still_resets_the_journal_and_says_it_started() -> None:
        script = bootstrap.script(Image(pip=["six"]), "skyward")

        assert "rm -f /opt/skyward/events.jsonl" in script
        assert "emit_phase started bootstrap" in script


def describe_the_include_path_file() -> None:
    @pytest.mark.local
    def it_inserts_the_directory_at_the_front_of_sys_path() -> None:
        assert bootstrap.pth("/x") == "import sys; sys.path.insert(0, '/x')\n"


AWS = Endpoint(url="https://s3.us-west-2.amazonaws.com", region="us-west-2")


def describe_the_volumes_phase() -> None:
    @pytest.mark.local
    def its_commands_are_valid_bash_with_volumes_of_both_types(tmp_path: Path) -> None:
        phase = bootstrap.mounts(
            (
                (Volume(bucket="datasets", mount="/data", prefix="imagenet/"), AWS),
                (Volume(bucket="runs", mount="/out", read_only=False, type="sync"), AWS),
                (Volume(bucket="runs", mount="/in", prefix="seed", type="sync"), Endpoint(url="https://r2.example", access_key="k", secret_key="s")),
            ),
        )
        _, _, commands = shlex.split(phase)
        path = tmp_path / "volumes.sh"
        path.write_text(commands)

        done = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True, timeout=10, check=False)

        assert done.returncode == 0, done.stderr

    @pytest.mark.local
    def a_synced_volume_needs_no_fuse() -> None:
        phase = bootstrap.mounts(((Volume(bucket="runs", mount="/out", type="sync"), AWS),))

        assert "geesefs" not in phase
        assert "fuse" not in phase

    @pytest.mark.local
    def a_synced_volume_signs_for_the_region_of_its_endpoint() -> None:
        phase = bootstrap.mounts(((Volume(bucket="runs", mount="/out", type="sync"), AWS),))

        assert "region = us-west-2" in phase

    @pytest.mark.local
    def every_copy_compares_upload_times_and_says_it_is_skyward() -> None:
        phase = bootstrap.mounts(((Volume(bucket="runs", mount="/out", read_only=False, type="sync"), AWS),))

        assert phase.count(" copy ") == phase.count(" copy --update --use-server-modtime --user-agent skyward/") == 3


SYNC_RUNNER = """
cat > /tmp/bootstrap.sh
curl -fsSL -o /tmp/rclone.deb "https://github.com/rclone/rclone/releases/download/@RCLONE@/rclone-@RCLONE@-linux-$(dpkg --print-architecture).deb"
dpkg -i /tmp/rclone.deb > /dev/null
mkdir -p /srv/bucket/runs /srv/bucket/seed
echo seeded > /srv/bucket/runs/a.txt
echo seeded > /srv/bucket/seed/a.txt
rclone serve s3 /srv --addr 127.0.0.1:9000 --auth-key key,secret > /tmp/serve.log 2>&1 &
sleep 3
bash /tmp/bootstrap.sh || { cat /opt/skyward/events.jsonl; exit 1; }
echo ours > /out/b.txt
echo ours > /in/b.txt
echo theirs > /tmp/c.txt
rclone copyto /tmp/c.txt ":s3,provider=Other,endpoint='http://127.0.0.1:9000',access_key_id=key,secret_access_key=secret,force_path_style=true:bucket/runs/c.txt"
sleep @SETTLE@
echo "bucket $(ls /srv/bucket/runs | tr '\\n' ' ')"
echo "out $(ls /out | tr '\\n' ' ')"
echo "seed $(ls /srv/bucket/seed | tr '\\n' ' ')"
pgrep -f "/opt/skyward/sync.sh [0-9]" > /dev/null && echo "before running" || echo "before stopped"
printf 'newer\\n' > /opt/skyward/sync.generation
sleep @STOP@
pgrep -f "/opt/skyward/sync.sh [0-9]" > /dev/null && echo "after running" || echo "after stopped"
"""


def describe_a_synced_volume_on_linux() -> None:
    @pytest.mark.compute
    @pytest.mark.xdist_group("bootstrap")
    def it_copies_the_bucket_in_then_keeps_both_sides_in_step_until_a_newer_generation_takes_over() -> None:
        served = Endpoint(url="http://127.0.0.1:9000", access_key="key", secret_key="secret", path_style=True)
        phase = bootstrap.mounts(
            (
                (Volume(bucket="bucket", mount="/out", prefix="runs", read_only=False, type="sync"), served),
                (Volume(bucket="bucket", mount="/in", prefix="seed", type="sync"), served),
            ),
        )
        runner = (
            SYNC_RUNNER.replace("@RCLONE@", bootstrap.RCLONE)
            .replace("@SETTLE@", str(2 * bootstrap.SYNC_INTERVAL + 5))
            .replace("@STOP@", str(2 * bootstrap.SYNC_INTERVAL + 5))
        )

        done = subprocess.run(
            ["docker", "run", "--rm", "-i", IMAGE, "bash", "-c", runner],
            input="\n".join((bootstrap.HEADER, phase, bootstrap.FOOTER)),
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )

        assert done.returncode == 0, done.stdout + done.stderr
        listed = {line.split(" ", 1)[0]: line.split(" ", 1)[1].split() for line in done.stdout.splitlines()}
        assert listed["bucket"] == ["a.txt", "b.txt", "c.txt"]
        assert listed["out"] == ["a.txt", "b.txt", "c.txt"]
        assert listed["seed"] == ["a.txt"]
        assert listed["before"] == ["running"]
        assert listed["after"] == ["stopped"]
