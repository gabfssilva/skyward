"""Base protocol for composable Rich components."""

from __future__ import annotations

from typing import Protocol

from rich.console import RenderableType


class Component(Protocol):
    """Protocol for composable Rich components.

    Components consume ViewModels and produce Rich renderables.
    They are stateless and purely functional.
    """

    def render(self, width: int) -> RenderableType:
        """Render component to Rich renderable.

        Args:
            width: Available terminal width for rendering.

        Returns:
            A Rich renderable (Text, Table, etc.)
        """
        ...
