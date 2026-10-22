"""The vocabulary the console and the app share: the palette, the badges, the bar, the names of the bootstrap steps.

Pure drawing: nothing here knows about the daemon or its events.
"""

from __future__ import annotations

import os

from rich.style import Style
from rich.text import Text

DIM = Style(color="bright_black")
WARNING_STYLE = "yellow"


def _detect_terminal_bg() -> tuple[int, int, int] | None:
    import select
    import sys
    import termios
    import tty

    try:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
    except (OSError, termios.error):
        return None

    try:
        tty.setraw(fd)
        os.write(sys.stdout.fileno(), b"\033]11;?\033\\")
        if not select.select([fd], [], [], 0.5)[0]:
            return None
        response = b""
        while select.select([fd], [], [], 0.1)[0]:
            response += os.read(fd, 1024)
        decoded = response.decode("latin-1")
        if "rgb:" not in decoded:
            return None
        parts = decoded.split("rgb:")[1].split("\033")[0].split("\\")[0].split("/")
        if len(parts) != 3:
            return None
        return int(parts[0][:2], 16), int(parts[1][:2], 16), int(parts[2][:2], 16)
    except Exception:
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _is_dark_background() -> bool:
    if (background := _detect_terminal_bg()) is not None:
        red, green, blue = background
        return 0.299 * red + 0.587 * green + 0.114 * blue < 128
    return os.environ.get("COLORFGBG", "").rpartition(";")[2] not in ("7", "15")


_DARK = _is_dark_background()
_BADGE_L = 0.55 if _DARK else 0.35
_BADGE_FG = "rgb(0,0,0)" if _DARK else "rgb(255,255,255)"
WARNING_STYLE = "yellow" if _DARK else "dark_orange"

_FIXED_BADGE_HUES: dict[str, tuple[float, float]] = {
    "skyward": (255.0, 0.50),
    "cluster": (150.0, 0.45),
    "error": (0.0, 0.60),
    "local": (0.0, 0.0),
    "queued": (30.0, 0.50),
    "running": (210.0, 0.60),
    "done": (120.0, 0.50),
    "failed": (0.0, 0.60),
    "connecting": (45.0, 0.45),
    "bootstrap": (45.0, 0.45),
    "provisioning": (45.0, 0.45),
    "ready": (120.0, 0.50),
    "scaling": (45.0, 0.45),
    "in sync": (120.0, 0.50),
    "drifted": (0.0, 0.60),
    "shutting down": (0.0, 0.0),
    "cost": (0.0, 0.0),
}


def _hsl_to_rgb(hue: float, saturation: float, lightness: float) -> tuple[int, int, int]:
    chroma = (1 - abs(2 * lightness - 1)) * saturation
    secondary = chroma * (1 - abs((hue / 60) % 2 - 1))
    match hue:
        case value if value < 60:
            red, green, blue = chroma, secondary, 0.0
        case value if value < 120:
            red, green, blue = secondary, chroma, 0.0
        case value if value < 180:
            red, green, blue = 0.0, chroma, secondary
        case value if value < 240:
            red, green, blue = 0.0, secondary, chroma
        case value if value < 300:
            red, green, blue = secondary, 0.0, chroma
        case _:
            red, green, blue = chroma, 0.0, secondary
    offset = lightness - chroma / 2
    return int((red + offset) * 255), int((green + offset) * 255), int((blue + offset) * 255)


def _make_badge(hue: float, saturation: float) -> Style:
    red, green, blue = _hsl_to_rgb(hue % 360, saturation, _BADGE_L)
    return Style(color=_BADGE_FG, bgcolor=f"rgb({red},{green},{blue})", bold=True)


_FIXED_BADGES = {name: _make_badge(hue, saturation) for name, (hue, saturation) in _FIXED_BADGE_HUES.items()}
_FIXED_BADGES["skyward"] = Style(
    color="rgb(0,0,0)" if _DARK else "rgb(255,255,255)",
    bgcolor="rgb(255,255,255)" if _DARK else "rgb(0,0,0)",
    bold=True,
)


def _stable_hash(label: str) -> int:
    import hashlib

    return int.from_bytes(hashlib.md5(label.encode()).digest()[:4], "big")


def _badge_style(label: str) -> Style:
    if label in _FIXED_BADGES:
        return _FIXED_BADGES[label]
    if label.startswith("~$"):
        return _FIXED_BADGES["cost"]
    return _make_badge((_stable_hash(label) * 137.508) % 360, 0.65)


def _fill_style(hue: float) -> Style:
    red, green, blue = _hsl_to_rgb(hue % 360, 0.65, _BADGE_L)
    return Style(color=f"rgb({red},{green},{blue})")


_BAR_WIDTH = 24


def _bar(completion: float) -> Text:
    """How much of it is done, as a bar that walks from the colour of `provisioning` to the colour of `ready`.

    The palette is the one the badges use, so a bar filling and a badge turning green
    are the same event said twice rather than two vocabularies on one screen.
    """
    done = min(max(completion, 0.0), 1.0)
    filled = round(done * _BAR_WIDTH)
    text = Text()
    text.append("━" * filled, style=_fill_style(45 + 75 * done))
    text.append("━" * (_BAR_WIDTH - filled), style=DIM)
    return text


_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_PHASE_LABELS = {
    "connecting": "ssh",
    "env": "set env",
    "apt": "apt install",
    "uv": "setup uv",
    "venv": "create venv",
    "deps": "install deps",
    "skyward": "install skyward",
    "volumes": "mount volumes",
    "worker": "start worker",
}


def _accelerator_label(name: str) -> str:
    return name.upper().replace("-", " ")


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    remainder = int(seconds % 60)
    return f"{minutes}m {remainder:02d}s" if minutes < 60 else f"{minutes // 60}h {minutes % 60:02d}m"
