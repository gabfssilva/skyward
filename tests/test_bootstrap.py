"""The bootstrap script: as text, and as bash that at least parses."""

from __future__ import annotations

import shutil
import subprocess
from typing import ClassVar

import pytest

from skyward.worker.plugins.plugin import Plugin
from skyward.shared.schemas import Image, MetricSpec, PipIndex
from skyward.worker import bootstrap

pytestmark = pytest.mark.unit


class Slicer(Plugin, frozen=True):
    """A plugin that only asks for a phase, so the postamble can be seen in the script."""

    kind: ClassVar[str] = "slicer"

    def bootstrap(self, image: Image, concurrency: int) -> tuple[str, ...]:
        return (bootstrap.phase("slice", f"echo splitting into {concurrency}"),)


def test_the_metrics_collectors_start_before_the_bootstrap_phases():
    """Telemetry brackets the whole bootstrap, so a machine says how it is while it comes up."""
    text = bootstrap.script(Image(), "skyward")
    assert "emit_metric" in text
    assert "_collect_cpu()" in text and "_collect_gpu_util()" in text
    assert text.index("start_metrics_daemon") < text.index("phase uv")


def test_apt_installs_curl_and_git_and_extras_before_uv():
    """The uv installer is curl, so curl has to be there first — the RunPod regression."""
    text = bootstrap.script(Image(apt=("ffmpeg",)), "skyward")
    assert "phase apt" in text
    assert "curl" in text and "git" in text and "ffmpeg" in text
    assert text.index("phase apt") < text.index("phase uv")


def test_default_metrics_stay_when_none_is_asked_for():
    text = bootstrap.script(Image(), "skyward")
    assert "_collect_cpu()" in text and "_collect_gpu_util()" in text


def test_metrics_override_replaces_the_built_in_collectors():
    text = bootstrap.script(Image(metrics=(MetricSpec("temp", "cat /sys/class/thermal/x", 5.0),)), "skyward")
    assert "_collect_temp()" in text
    assert "_collect_cpu()" not in text


def test_shell_vars_resolve_and_reach_the_deps_phase():
    text = bootstrap.script(Image(shell_vars={"CUDA": "nvidia-smi --query"}, pip=("torch",)), "skyward")
    assert "phase shell_vars" in text
    assert "CUDA" in text and "nvidia-smi --query" in text
    assert text.index("phase shell_vars") < text.index("phase deps")
    deps = text.split("phase deps", 1)[1]
    assert bootstrap.VARS in deps


def test_scoped_pip_index_is_written_as_a_uv_source():
    text = bootstrap.script(
        Image(pip=("torch",), pip_indexes=(PipIndex("https://download.pytorch.org/whl/cpu", ("torch",)),)),
        "skyward",
    )
    assert "https://download.pytorch.org/whl/cpu" in text
    assert "explicit = true" in text
    assert "[tool.uv.sources]" in text
    assert 'torch = { index =' in text
    assert "uv add" in text


def test_a_plugins_phase_lands_after_the_image_and_before_the_footer():
    text = bootstrap.script(Image(), "skyward", (Slicer(),), 4)
    assert "phase slice" in text
    assert "splitting into 4" in text
    assert text.index("phase skyward") < text.index("phase slice") < text.index(bootstrap.FOOTER)


def test_no_plugins_is_byte_identical_to_the_bare_call():
    """The new parameters default, so the no-plugin script is exactly what it was."""
    assert bootstrap.script(Image(), "skyward") == bootstrap.script(Image(), "skyward", (), 1)
    assert "phase slice" not in bootstrap.script(Image(), "skyward")


def test_phase_joins_with_and_and_quotes_as_one_word():
    built = bootstrap.phase("deps", "cd /opt", "uv pip install torch")
    assert built == "phase deps 'cd /opt && uv pip install torch'"


@pytest.mark.parametrize(
    "image",
    [
        Image(),
        Image(apt=("ffmpeg", "libsndfile1")),
        Image(pip=("torch", "numpy"), shell_vars={"CUDA": "nvidia-smi -L | head -1"}),
        Image(metrics=(MetricSpec("temp", "sensors -u | awk 'NR==1'", 5.0),)),
        Image(pip=("torch",), pip_indexes=(PipIndex("https://example.com/whl", ("torch",)),)),
    ],
)
def test_the_generated_script_is_valid_bash(image: Image):
    text = bootstrap.script(image, "skyward")
    bash = shutil.which("bash")
    assert bash, "bash is required for this test"

    result = subprocess.run([bash, "-n"], input=text, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
