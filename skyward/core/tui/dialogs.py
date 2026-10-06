"""The questions ``sky app`` asks before it acts: confirm, resize, and a line of text.

Each is a modal screen that answers through :meth:`~textual.screen.Screen.dismiss`,
so the caller receives the answer in a callback and the screen underneath keeps
repainting. They are keyboard-only by design — tab moves between the fields and
the buttons, enter presses the focused one, escape cancels — and coloured with
theme variables alone, so they read on the light theme and the dark one. The
button in focus is underlined, not inverted: its label keeps its colour.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static


def _css(name: str) -> str:
    return f"""
    {name} {{ align: center middle; }}
    {name} > Vertical {{
        width: 60;
        height: auto;
        padding: 1 2;
        border: round $primary;
        background: $surface;
        color: $text;
    }}
    {name} .question {{ width: 1fr; height: auto; color: $error; text-style: bold; }}
    {name} .note {{ width: 1fr; height: auto; color: $text-muted; margin-top: 1; }}
    {name} .problem {{ width: 1fr; height: auto; color: $error; }}
    {name} .buttons {{ height: auto; margin-top: 1; align-horizontal: right; }}
    {name} .buttons > Button {{ margin-left: 1; }}
    {name} .buttons > Button:focus {{ text-style: bold underline; }}
    {name} Input {{ margin-top: 1; }}
    """


class Confirm(ModalScreen[bool]):
    """A question in red, and two buttons; focus starts on ``cancel``."""

    DEFAULT_CSS = _css("Confirm")
    BINDINGS = [Binding("escape", "cancel", "cancel")]

    def __init__(self, question: str, confirm: str, note: str = "") -> None:
        super().__init__()
        self._question = question
        self._confirm = confirm
        self._note = note

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(self._question, classes="question")
            if self._note:
                yield Label(self._note, classes="note")
            with Horizontal(classes="buttons"):
                yield Button(self._confirm, id="confirm", variant="error")
                yield Button("cancel", id="cancel")

    def on_mount(self) -> None:
        self.query_one("#cancel", Button).focus()

    def on_button_pressed(self, message: Button.Pressed) -> None:
        self.dismiss(message.button.id == "confirm")

    def action_cancel(self) -> None:
        self.dismiss(False)


class Scale(ModalScreen[tuple[int, int] | None]):
    """Two bounds; an empty field keeps the current one, unchanged bounds answer ``None``."""

    DEFAULT_CSS = _css("Scale")
    BINDINGS = [Binding("escape", "cancel", "cancel")]

    def __init__(self, minimum: int, maximum: int) -> None:
        super().__init__()
        self._minimum = minimum
        self._maximum = maximum

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("scale", classes="question")
            yield Input(placeholder=f"min {self._minimum}", id="min")
            yield Input(placeholder=f"max {self._maximum}", id="max")
            yield Static("", id="problem", classes="problem")
            with Horizontal(classes="buttons"):
                yield Button("apply", id="apply", variant="primary")
                yield Button("cancel", id="cancel")

    def on_mount(self) -> None:
        self.query_one("#min", Input).focus()

    def on_input_submitted(self) -> None:
        self._apply()

    def on_button_pressed(self, message: Button.Pressed) -> None:
        if message.button.id == "apply":
            self._apply()
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _apply(self) -> None:
        try:
            minimum = int(self.query_one("#min", Input).value.strip() or self._minimum)
            maximum = int(self.query_one("#max", Input).value.strip() or self._maximum)
        except ValueError:
            self._reject()
            return
        if minimum > maximum:
            self._reject()
            return
        self.dismiss(None if (minimum, maximum) == (self._minimum, self._maximum) else (minimum, maximum))

    def _reject(self) -> None:
        self.query_one("#problem", Static).update("whole numbers, min <= max")


class Ask(ModalScreen[str | None]):
    """One line of text; enter answers with it stripped, possibly empty."""

    DEFAULT_CSS = _css("Ask")
    BINDINGS = [Binding("escape", "cancel", "cancel")]

    def __init__(self, label: str, value: str = "") -> None:
        super().__init__()
        self._label = label
        self._value = value

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(self._label, classes="question")
            yield Input(self._value, id="text")

    def on_mount(self) -> None:
        self.query_one("#text", Input).focus()

    def on_input_submitted(self, message: Input.Submitted) -> None:
        self.dismiss(message.value.strip())

    def action_cancel(self) -> None:
        self.dismiss(None)


__all__ = ["Ask", "Confirm", "Scale"]
