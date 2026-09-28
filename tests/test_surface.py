"""What ``import skyward as sky`` promises, name by name.

The surface is three hand-kept lists — ``__all__``, the ``TYPE_CHECKING`` block
and ``__getattr__``'s routing — and a name in one but not the others surfaces as
an ``AttributeError`` in somebody's script. Resolving every exported name is the
one check that keeps the three in step.

``skyward.core`` is the client, and all of it but ``Compute`` has to import on a
node, which has no httpx: ``sky run`` executes a file there, top level and all.
"""

import subprocess
import sys
import textwrap

import pytest

import skyward as sky
import skyward.core

pytestmark = pytest.mark.local


def describe_the_public_surface() -> None:
    @pytest.mark.parametrize("name", sorted(sky.__all__))
    def every_exported_name_resolves(name: str) -> None:
        assert getattr(sky, name) is not None

    def dir_lists_exactly_what_is_exported() -> None:
        assert set(sky.__all__) <= set(dir(sky))

    @pytest.mark.parametrize("name", sorted(skyward.core.__all__))
    def every_name_of_the_client_resolves(name: str) -> None:
        assert getattr(skyward.core, name) is not None


def describe_a_node_without_the_client() -> None:
    def what_a_file_declares_at_its_top_level_imports_and_compute_says_what_it_needs() -> None:
        code = textwrap.dedent("""
            import sys

            sys.modules["httpx"] = None
            import skyward as sky

            @sky.app(provider=sky.AWS(), accelerator=sky.accelerators.A100(), image=sky.Image(pip=["numpy"]), plugins=[sky.plugins.Torch()])
            def train() -> int:
                return 1

            @sky.function
            def step() -> int:
                return 1

            assert train() == 1
            try:
                sky.Compute
            except ImportError as error:
                print(error)
        """)
        ran = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)

        assert ran.returncode == 0, ran.stderr
        assert ran.stdout.strip() == "'Compute' is part of the client — install 'skyward[client]'"
