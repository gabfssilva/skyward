import pytest

from skyward.api.spec import Nodes


class TestNodes:
    def test_fixed(self) -> None:
        n = Nodes(min=4)
        assert n.min == 4
        assert n.max is None
        assert n.desired is None
        assert not n.auto_scaling

    def test_autoscaling(self) -> None:
        n = Nodes(min=2, max=8)
        assert n.min == 2
        assert n.max == 8
        assert n.auto_scaling

    def test_desired(self) -> None:
        n = Nodes(min=4, desired=2)
        assert n.desired == 2
        assert not n.auto_scaling

    def test_all_fields(self) -> None:
        n = Nodes(min=4, desired=2, max=8)
        assert n.min == 4
        assert n.desired == 2
        assert n.max == 8
        assert n.auto_scaling

    def test_min_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="min must be >= 1"):
            Nodes(min=0)

    def test_max_must_be_gte_min(self) -> None:
        with pytest.raises(ValueError, match="max .* must be >= min"):
            Nodes(min=4, max=2)

    def test_desired_must_be_lte_min_fixed(self) -> None:
        with pytest.raises(ValueError, match="desired .* must be <= min"):
            Nodes(min=2, desired=4)

    def test_desired_can_exceed_min_with_max(self) -> None:
        n = Nodes(min=1, desired=2, max=3)
        assert n.desired == 2

    def test_desired_must_be_lte_max(self) -> None:
        with pytest.raises(ValueError, match="desired .* must be <= max"):
            Nodes(min=1, desired=5, max=3)

    def test_desired_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="desired must be >= 1"):
            Nodes(min=4, desired=0)
