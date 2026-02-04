from skyward.distributed.types import Consistency


def test_consistency_literal():
    """Consistency type accepts valid values."""
    strong: Consistency = "strong"
    eventual: Consistency = "eventual"
    assert strong == "strong"
    assert eventual == "eventual"
