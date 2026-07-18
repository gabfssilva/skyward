import pytest

import skyward.metrics as metrics
from skyward.protocol.schemas import MetricSpec

pytestmark = pytest.mark.unit


def test_cpu():
    spec = metrics.CPU()
    assert isinstance(spec, MetricSpec)
    assert spec.name == "cpu"
    assert spec.interval == 2
    assert "top -bn2" in spec.command


def test_memory():
    spec = metrics.Memory(interval=0.5)
    assert spec.name == "mem"
    assert spec.interval == 0.5
    assert "free" in spec.command


def test_memory_used():
    spec = metrics.MemoryUsed()
    assert spec.name == "mem_used_mb"
    assert "$3/1024" in spec.command


def test_memory_total():
    spec = metrics.MemoryTotal()
    assert spec.name == "mem_total_mb"
    assert spec.interval == 60.0
    assert "$2/1024" in spec.command


def test_gpu_aggregate():
    spec = metrics.GPU()
    assert spec.name == "gpu_util"
    assert spec.interval == 3
    assert "utilization.gpu" in spec.command
    assert "-i " not in spec.command


def test_gpu_indexed():
    spec = metrics.GPU(index=2, interval=1)
    assert spec.name == "gpu_util_2"
    assert spec.interval == 1
    assert "-i 2 " in spec.command


def test_gpu_memory():
    spec = metrics.GPUMemory()
    assert spec.name == "gpu_mem_mb"
    assert "memory.used" in spec.command


def test_gpu_memory_total():
    spec = metrics.GPUMemoryTotal()
    assert spec.name == "gpu_mem_total_mb"
    assert spec.interval == 60.0
    assert "memory.total" in spec.command


def test_gpu_temp():
    spec = metrics.GPUTemp(index=0)
    assert spec.name == "gpu_temp_0"
    assert "temperature.gpu" in spec.command


def test_disk():
    spec = metrics.Disk("/data")
    assert spec.name == "disk_data"
    assert spec.interval == 5.0
    assert "df /data" in spec.command


def test_disk_root():
    assert metrics.Disk("/").name == "disk_root"


def test_network_rx():
    spec = metrics.NetworkRx("ens5")
    assert spec.name == "net_rx_ens5"
    assert "rx_bytes" in spec.command


def test_network_tx():
    spec = metrics.NetworkTx()
    assert spec.name == "net_tx_eth0"
    assert "tx_bytes" in spec.command


def test_custom():
    spec = metrics.Custom("open_fds", "ls /proc/self/fd | wc -l", interval=5)
    assert spec.name == "open_fds"
    assert spec.command == "ls /proc/self/fd | wc -l"
    assert spec.interval == 5


def test_default_set():
    specs = metrics.Default()
    assert all(isinstance(s, MetricSpec) for s in specs)
    names = tuple(s.name for s in specs)
    assert names == (
        "cpu",
        "mem",
        "mem_used_mb",
        "mem_total_mb",
        "gpu_util",
        "gpu_mem_mb",
        "gpu_mem_total_mb",
        "gpu_temp",
    )


def test_default_intervals():
    specs = metrics.Default(cpu_interval=0.5, gpu_interval=2.0)
    by_name = {s.name: s for s in specs}
    assert by_name["cpu"].interval == 0.5
    assert by_name["gpu_util"].interval == 2.0
