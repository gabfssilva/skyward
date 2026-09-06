"""The renderables ``sky app`` draws, against the views they are drawn from."""

from __future__ import annotations

from skyward.core.tui import tail
from skyward.core.view import ComputeView, NodeView


def describe_the_output_pane() -> None:
    def the_last_lines_come_from_every_node_in_the_order_spoken() -> None:
        view = ComputeView(
            id="cmp_1",
            nodes=(NodeView("nod_a", rank=0), NodeView("nod_b", rank=1)),
            tail=(("nod_a", "a1"), ("nod_b", "b1"), ("nod_a", "a2"), ("nod_b", "b2"), ("nod_a", "a3")),
        )

        assert tail(view, 3).plain == "[node 0] a2\n[node 1] b2\n[node 0] a3"
