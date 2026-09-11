"""The metrics collectors a bootstrap starts, as the shell that runs them sees them."""

import json
import subprocess
from pathlib import Path

import pytest

from skyward.shared.schemas import MetricSpec
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
    def it_is_valid_bash_with_custom_specs(tmp_path: Path) -> None:
        specs = (
            MetricSpec(name="load", command="cut -d' ' -f1 /proc/loadavg", interval=1),
            MetricSpec(name="disk", command="df / | awk 'NR==2 {print $5}' | tr -d %", interval=2.5),
        )

        done = _syntax(tmp_path, bootstrap.metrics(specs))

        assert done.returncode == 0, done.stderr


def describe_the_builtin_loop_on_linux() -> None:
    @pytest.mark.compute
    @pytest.mark.xdist_group("bootstrap")
    def it_reports_cpu_and_memory_and_stops_when_a_newer_generation_takes_over() -> None:
        done = subprocess.run(
            ["docker", "run", "--rm", "-i", IMAGE, "bash", "-c", RUNNER],
            input=bootstrap.HEADER + "\n" + bootstrap.metrics(None) + "\n",
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

        assert done.returncode == 0, done.stderr
        metrics, counts = done.stdout.split("---\n")
        records = [json.loads(line) for line in metrics.splitlines()]
        names = {record["name"] for record in records}
        assert {"cpu", "mem_used_mb", "mem_total_mb"} <= names
        assert all(set(record) == {"type", "name", "value"} and isinstance(record["value"], int | float) for record in records)
        first, second = counts.split()
        assert first == second
