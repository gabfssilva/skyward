from pathlib import Path

import pytest

import skyward as sky

pytestmark = pytest.mark.e2e


@sky.function
def standalone_info() -> tuple[int, int]:
    info = sky.instance_info()
    return info.rank, info.nodes


@sky.function
def standalone_collection_error() -> str:
    try:
        sky.dict("unavailable")
    except RuntimeError as exc:
        return str(exc)
    raise AssertionError("distributed collections must be unavailable in standalone mode")


def test_standalone_dispatch_reaches_every_independent_worker(tmp_path: Path) -> None:
    with sky.Compute(
        provider=sky.Container(),
        nodes=2,
        options=sky.Options(cluster=False),
        database=tmp_path / "skyward.sqlite",
    ) as compute:
        assert sorted(standalone_info() @ compute) == [(0, 2), (1, 2)]
        assert "distributed collections" in (standalone_collection_error() >> compute)
