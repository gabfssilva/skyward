"""Where a node gets casty from when the daemon's is unpublished.

A casty installed from an index is one a node installs from that index too. One
installed from a checkout exists nowhere else, so the daemon ships the checkout's
manylinux wheels of that version — or refuses before a machine installs a casty the
daemon cannot speak to.
"""

import json
from importlib.metadata import PathDistribution
from pathlib import Path

import pytest

from skyward.server.application import source

pytestmark = pytest.mark.local


def installed(root: Path, origin: dict[str, object] | None) -> PathDistribution:
    """A casty 0.30.0 as ``importlib.metadata`` sees it, installed from ``origin`` — or from an index, without one."""
    info = root / "casty-0.30.0.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: casty\nVersion: 0.30.0\n")
    if origin is not None:
        (info / "direct_url.json").write_text(json.dumps(origin))
    return PathDistribution(info)


def wheel(dist: Path, name: str) -> None:
    dist.mkdir(parents=True, exist_ok=True)
    (dist / name).write_bytes(name.encode())


def describe_the_casty_a_node_installs() -> None:
    def from_an_index_is_nothing_to_ship(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(source, "distribution", lambda _: installed(tmp_path / "site", None))

        assert source.checkout() == ()

    def from_a_checkout_is_its_linux_wheels_of_that_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        checkout = tmp_path / "my casty"
        for name in (
            "casty-0.30.0-cp312-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
            "casty-0.30.0-cp312-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
            "casty-0.30.0-cp312-abi3-macosx_11_0_arm64.whl",
            "casty-0.29.0-cp312-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        ):
            wheel(checkout / "dist", name)
        origin = {"url": checkout.as_uri(), "dir_info": {"editable": True}}
        monkeypatch.setattr(source, "distribution", lambda _: installed(tmp_path / "site", origin))

        shipped = sorted(shipped.name for shipped in source.checkout())

        assert shipped == [
            "casty-0.30.0-cp312-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
            "casty-0.30.0-cp312-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        ], "one per architecture, for the node to pick from; never another version, nor a wheel no node can run"

    def from_a_checkout_without_them_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        checkout = tmp_path / "casty"
        wheel(checkout / "dist", "casty-0.30.0-cp312-abi3-macosx_11_0_arm64.whl")
        origin = {"url": checkout.as_uri(), "dir_info": {}}
        monkeypatch.setattr(source, "distribution", lambda _: installed(tmp_path / "site", origin))

        with pytest.raises(RuntimeError, match="no manylinux wheel"):
            source.checkout()

    async def is_found_by_the_install_beside_the_skyward_wheel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        shipped = (source.Wheel(name="casty-0.30.0-cp312-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl", data=b""),)
        monkeypatch.setattr(source, "build", lambda: (source.Wheel(name="skyward-1.0-py3-none-any.whl", data=b""),))
        monkeypatch.setattr(source, "checkout", lambda: shipped)

        resolved = await source.resolve("local")

        assert resolved.arguments == (
            "--find-links",
            source.SKYWARD_DIR,
            "--reinstall-package",
            "casty",
            f"{source.SKYWARD_DIR}/skyward-1.0-py3-none-any.whl",
        )
        assert [wheel.name for wheel in resolved.wheels] == ["skyward-1.0-py3-none-any.whl", *(wheel.name for wheel in shipped)]
