from skyward.accelerators.spec import Accelerator


class TestAcceleratorFractionalCount:
    def test_fractional_count(self) -> None:
        accel = Accelerator("H100", count=0.5)
        assert accel.count == 0.5

    def test_str_fractional(self) -> None:
        assert str(Accelerator("H100", count=0.5)) == "0.5xH100"

    def test_str_integer_no_trailing_zero(self) -> None:
        assert str(Accelerator("H100", count=2)) == "2xH100"

    def test_str_single_omits_count(self) -> None:
        assert str(Accelerator("H100", count=1)) == "H100"

    def test_str_float_one_omits_count(self) -> None:
        assert str(Accelerator("H100", count=1.0)) == "H100"

    def test_with_count_float(self) -> None:
        accel = Accelerator("H100", count=1)
        fractional = accel.with_count(0.5)
        assert fractional.count == 0.5
        assert str(fractional) == "0.5xH100"

    def test_repr_fractional(self) -> None:
        accel = Accelerator("H100", count=0.5)
        assert "count=0.5" in repr(accel)

    def test_from_name_fractional(self) -> None:
        accel = Accelerator.from_name("H100", count=0.5)
        assert accel.count == 0.5
        assert accel.memory == "80GB"


from skyward.api.spec import PoolSpec, Nodes


class TestPoolSpecFractionalCount:
    def test_accelerator_count_returns_float(self) -> None:
        spec = PoolSpec(
            nodes=Nodes(min=1),
            accelerator=Accelerator("H100", count=0.5),
            region="us-east-1",
        )
        assert spec.accelerator_count == 0.5

    def test_accelerator_count_no_accelerator(self) -> None:
        spec = PoolSpec(
            nodes=Nodes(min=1),
            accelerator=None,
            region="us-east-1",
        )
        assert spec.accelerator_count == 0
