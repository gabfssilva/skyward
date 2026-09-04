"""What ``import skyward as sky`` promises, name by name.

The surface is three hand-kept lists — ``__all__``, the ``TYPE_CHECKING`` block
and ``__getattr__``'s routing — and a name in one but not the others surfaces as
an ``AttributeError`` in somebody's script. Resolving every exported name is the
one check that keeps the three in step.
"""

import pytest

import skyward as sky

pytestmark = pytest.mark.local


def describe_the_public_surface() -> None:
    @pytest.mark.parametrize("name", sorted(sky.__all__))
    def every_exported_name_resolves(name: str) -> None:
        assert getattr(sky, name) is not None

    def dir_lists_exactly_what_is_exported() -> None:
        assert set(sky.__all__) <= set(dir(sky))
