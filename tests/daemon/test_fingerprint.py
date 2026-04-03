from skyward.accelerators import Accelerator
from skyward.api.spec import Image, Spec
from skyward.daemon.fingerprint import compute_fingerprint


class _FakeProvider:
    """Minimal ProviderConfig stub for testing."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def type(self) -> str:
        return self._name

    async def create_provider(self):  # noqa: ANN201
        raise NotImplementedError

    def default_options(self):  # noqa: ANN201
        return None


class TestComputeFingerprint:
    def test_deterministic(self) -> None:
        """Same spec always produces same fingerprint."""
        spec = Spec(provider=_FakeProvider("aws"), accelerator=Accelerator("A100"), region="us-east-1")
        assert compute_fingerprint(spec) == compute_fingerprint(spec)

    def test_slug_format(self) -> None:
        """Fingerprint is human-readable slug with hash suffix."""
        spec = Spec(provider=_FakeProvider("aws"), accelerator=Accelerator("A100"), region="us-east-1")
        fp = compute_fingerprint(spec)
        assert fp.startswith("aws-A100-us-east-1-")
        # 6-char hex suffix
        suffix = fp.split("-")[-1]
        assert len(suffix) == 6
        int(suffix, 16)  # valid hex

    def test_no_accelerator_uses_cpu(self) -> None:
        spec = Spec(provider=_FakeProvider("aws"), region="us-east-1")
        fp = compute_fingerprint(spec)
        assert fp.startswith("aws-cpu-us-east-1-")

    def test_different_image_different_fingerprint(self) -> None:
        base = Spec(provider=_FakeProvider("aws"), accelerator=Accelerator("A100"), region="us-east-1")
        with_torch = Spec(
            provider=_FakeProvider("aws"),
            accelerator=Accelerator("A100"),
            region="us-east-1",
            image=Image(pip=["torch"]),
        )
        assert compute_fingerprint(base) != compute_fingerprint(with_torch)

    def test_different_nodes_same_fingerprint(self) -> None:
        """Nodes are operational, not identity."""
        spec4 = Spec(provider=_FakeProvider("aws"), accelerator=Accelerator("A100"), region="us-east-1", nodes=4)
        spec8 = Spec(provider=_FakeProvider("aws"), accelerator=Accelerator("A100"), region="us-east-1", nodes=8)
        assert compute_fingerprint(spec4) == compute_fingerprint(spec8)

    def test_different_provider_different_fingerprint(self) -> None:
        aws = Spec(provider=_FakeProvider("aws"), accelerator=Accelerator("A100"), region="us-east-1")
        gcp = Spec(provider=_FakeProvider("gcp"), accelerator=Accelerator("A100"), region="us-east-1")
        assert compute_fingerprint(aws) != compute_fingerprint(gcp)

    def test_no_region_uses_default(self) -> None:
        spec = Spec(provider=_FakeProvider("aws"), accelerator=Accelerator("A100"))
        fp = compute_fingerprint(spec)
        assert "default" in fp

    def test_different_allocation_same_fingerprint(self) -> None:
        """Allocation strategy is operational, not identity."""
        spot = Spec(
            provider=_FakeProvider("aws"), accelerator=Accelerator("A100"), region="us-east-1", allocation="spot"
        )
        demand = Spec(
            provider=_FakeProvider("aws"),
            accelerator=Accelerator("A100"),
            region="us-east-1",
            allocation="on-demand",
        )
        assert compute_fingerprint(spot) == compute_fingerprint(demand)
