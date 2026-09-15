"""The metrics collectors a bootstrap starts, as the shell that runs them sees them."""

import json
import subprocess
import time
from pathlib import Path
from typing import get_args

import msgspec
import pytest

import skyward as sky
from skyward.shared.schemas import READINGS, Image, MetricSpec, Reading
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
