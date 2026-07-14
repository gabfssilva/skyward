"""What the pool tells the person watching."""

from skyward2.sdk.console import render


def test_what_the_code_on_a_node_printed_is_the_point():
    assert render("node.console", {"node": "nod_1", "content": "epoch 3"}) == "nod_1 │ epoch 3"


def test_the_machines_say_where_they_got_to():
    assert render("node.ready", {"node": "nod_1"}) == "nod_1 │ ready"
    assert render("compute.ready", {"compute": "cmp_1"}) == "cmp_1 │ ready"


def test_a_failure_says_why():
    assert render("node.failed", {"node": "nod_1", "error": "no such package: torhc"}) == "nod_1 │ node.failed: no such package: torhc"


def test_a_task_gets_no_line():
    """The caller is holding the result, or the exception. Both say more than a line would."""
    assert render("task.succeeded", {"compute": "cmp_1", "task": "tsk_1"}) is None
