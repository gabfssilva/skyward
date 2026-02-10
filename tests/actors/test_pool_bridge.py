from skyward.actors.node import node_actor


def test_node_actor_importable():
    assert callable(node_actor)
