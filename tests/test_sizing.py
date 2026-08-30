"""How many machines a compute asks for, and when it stops asking.

``initial`` is the size a pool opens at, not a target it is held to: the request is
made once per generation, and a machine it asked for and never got is not asked for
again. What holds the pool up afterwards is ``min`` — the same count that decides
whether it is ready — and what makes it move at all afterwards is ``max``.
"""

import msgspec
import pytest

from skyward.core.spec import bounds as spelled
from skyward.server.application.mock import COMPUTE, NODE, SPEC
from skyward.server.application.reconciler import demand
from skyward.shared.schemas import Compute, Node, NodeBounds, NodeState, PluginRef

pytestmark = pytest.mark.local


def pool(nodes: NodeBounds, *plugins: str, slots: int = 1, generation: int = 3, deleted: bool = False) -> Compute:
    """A compute of that size, at that generation, and nothing else worth reading."""
    spec = msgspec.structs.replace(
        SPEC,
        nodes=nodes,
        worker=msgspec.structs.replace(SPEC.worker, concurrency=slots),
        plugins=tuple(PluginRef(kind=kind) for kind in plugins),
        desired="deleted" if deleted else "running",
    )
    return msgspec.structs.replace(COMPUTE, spec=spec, generation=generation)


def rows(*states: NodeState, generation: int = 3) -> tuple[Node, ...]:
    """One node row per state named, in the generation that asked for them."""
    return tuple(
        msgspec.structs.replace(NODE, id=f"nod_{index}", rank=index, state=state, generation=generation)
        for index, state in enumerate(states)
    )


def describe_a_pool_opening_at_the_size_it_was_given() -> None:
    def it_asks_for_all_of_it_at_once() -> None:
        buy, _ = demand(pool(NodeBounds(initial=5, min=2)), (), load=0)

        assert buy == 5, "the opening request is the whole size, not the floor it may live at"

    def it_asks_for_nothing_more_while_the_machines_are_on_their_way() -> None:
        buy, _ = demand(pool(NodeBounds(initial=5, min=2)), rows("requested", "provisioning", "connecting", "bootstrapping", "ready"), load=0)

        assert buy == 0, "a row is a machine already being bought"

    def it_does_not_ask_again_for_the_ones_that_never_came_up() -> None:
        standing = rows("ready", "ready", "ready", "failed", "failed")

        buy, spare = demand(pool(NodeBounds(initial=5, min=2)), standing, load=0)

        assert buy == 0, "three is not short of five: the request was made and this is what it got"
        assert spare <= 0, "and three is not two too many either"

    def it_buys_again_only_once_it_falls_under_the_floor() -> None:
        standing = rows("ready", "deleted", "deleted", "failed", "failed")

        buy, _ = demand(pool(NodeBounds(initial=5, min=2)), standing, load=0)

        assert buy == 1, "under the floor is the one thing the pool does not tolerate"


def describe_a_pool_given_one_number() -> None:
    def it_is_held_to_it_because_the_floor_is_the_number_it_named() -> None:
        buy, _ = demand(pool(NodeBounds(initial=4)), rows("ready", "ready", "ready", "lost"), load=0)

        assert buy == 1, "nodes=4 asked for four and never said it would live on fewer"


def describe_a_pool_asked_for_a_different_size() -> None:
    def it_grows_what_is_already_standing_rather_than_starting_over() -> None:
        buy, _ = demand(pool(NodeBounds(initial=8), generation=4), rows("ready", "ready", "ready", generation=3), load=0)

        assert buy == 5, "a resize from three to eight asks for five"

    def it_gives_back_what_the_new_size_does_not_want() -> None:
        buy, spare = demand(pool(NodeBounds(initial=2), generation=4), rows("ready", "ready", "ready", "ready", "ready", generation=3), load=0)

        assert (buy, spare) == (0, 3), "the machines over the new size are the ones to drain"


def describe_a_pool_with_a_ceiling() -> None:
    def it_opens_at_the_size_it_was_given_whatever_the_load_says() -> None:
        buy, _ = demand(pool(NodeBounds(initial=4, min=2, max=16)), (), load=0)

        assert buy == 4, "an empty queue is not a reason to open smaller than asked"

    def it_buys_for_the_work_that_is_waiting() -> None:
        buy, _ = demand(pool(NodeBounds(initial=2, min=2, max=8)), rows("ready", "ready"), load=6)

        assert buy == 4, "six tasks, one slot each, and two machines to run them on"

    def it_gives_back_what_the_work_no_longer_needs() -> None:
        standing = rows("ready", "ready", "ready", "ready", "ready", "ready")

        _, spare = demand(pool(NodeBounds(initial=2, min=2, max=8)), standing, load=0)

        assert spare == 4, "an idle elastic pool falls to its floor"

    def it_stops_at_the_ceiling_however_long_the_queue_is() -> None:
        buy, _ = demand(pool(NodeBounds(initial=2, min=2, max=8)), rows("ready", "ready"), load=1000)

        assert buy == 6, "max is the ceiling, and a queue is not an argument against it"


def describe_a_pool_running_a_collective() -> None:
    def it_is_held_to_the_size_it_opened_at_rather_than_the_floor() -> None:
        buy, _ = demand(pool(NodeBounds(initial=4, min=1), "torch"), rows("ready", "ready", "ready", "failed"), load=0)

        assert buy == 1, "a world of four that opened on three is a rendezvous one rank short"


def describe_a_pool_being_deleted() -> None:
    def it_buys_nothing_and_gives_everything_back() -> None:
        standing = rows("ready", "ready", "requested")

        assert demand(pool(NodeBounds(initial=5, min=2), deleted=True), standing, load=100) == (0, 3)


def describe_the_ways_a_size_is_spelled() -> None:
    def a_count_is_the_size_the_pool_opens_at_and_stays_at() -> None:
        assert spelled(4) == NodeBounds(initial=4)

    def a_range_opens_at_its_floor_and_grows_into_the_rest() -> None:
        assert spelled((2, 8)) == NodeBounds(initial=2, min=2, max=8)
