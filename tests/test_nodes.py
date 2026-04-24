import pytest

from skyward.api.spec import Nodes


class TestNodes:
    def test_autoscaling(self) -> None:
        n = Nodes(desired=2, max=8)
        assert n.desired == 2
        assert n.max == 8
        assert n.auto_scaling

    def test_min(self) -> None:
        n = Nodes(desired=4, min=2)
        assert n.min == 2
        assert not n.auto_scaling

    def test_desired_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="desired must be >= 1"):
            Nodes(desired=0)

    def test_max_must_be_gte_desired(self) -> None:
        with pytest.raises(ValueError, match="max .* must be >= desired"):
            Nodes(desired=4, max=2)

    def test_min_must_be_lte_desired_fixed(self) -> None:
        with pytest.raises(ValueError, match="min .* must be <= desired"):
            Nodes(desired=2, min=4)

    def test_min_can_exceed_desired_with_max(self) -> None:
        n = Nodes(desired=1, min=2, max=3)
        assert n.min == 2

    def test_min_must_be_lte_max(self) -> None:
        with pytest.raises(ValueError, match="min .* must be <= max"):
            Nodes(desired=1, min=5, max=3)

    def test_min_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="min must be >= 1"):
            Nodes(desired=4, min=0)


class TestFromSpec:
    def test_int_becomes_fixed_nodes(self) -> None:
        result = Nodes.from_spec(3)
        assert result == Nodes(desired=3)
        assert result.auto_scaling is False

    def test_tuple_becomes_elastic_nodes(self) -> None:
        result = Nodes.from_spec((2, 5))
        assert result == Nodes(desired=2, max=5)
        assert result.auto_scaling is True

    def test_nodes_passes_through(self) -> None:
        spec = Nodes(desired=4, min=2, max=8)
        assert Nodes.from_spec(spec) is spec

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(TypeError, match="NodeSpec"):
            Nodes.from_spec("4")  # type: ignore[arg-type]
