import pytest

from skyward.api.spec import Nodes


class TestNodes:
    def test_fixed(self) -> None:
        n = Nodes(desired=4)
        assert n.desired == 4
        assert n.max is None
        assert n.min is None
        assert not n.auto_scaling

    def test_autoscaling(self) -> None:
        n = Nodes(desired=2, max=8)
        assert n.desired == 2
        assert n.max == 8
        assert n.auto_scaling

    def test_min(self) -> None:
        n = Nodes(desired=4, min=2)
        assert n.min == 2
        assert not n.auto_scaling

    def test_all_fields(self) -> None:
        n = Nodes(desired=4, min=2, max=8)
        assert n.desired == 4
        assert n.min == 2
        assert n.max == 8
        assert n.auto_scaling

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
