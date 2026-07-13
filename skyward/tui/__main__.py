"""Entry point: ``python -m skyward.tui`` runs the simulated demo."""

from __future__ import annotations

import argparse

from skyward.tui.app import run


def main() -> None:
    """Parse CLI flags and launch the simulated TUI."""
    parser = argparse.ArgumentParser(prog="skyward.tui", description="Skyward cluster monitor (demo)")
    parser.add_argument("--dark", action="store_true", help="start in the dark theme")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between ticks")
    parser.add_argument("--seed", type=int, default=7, help="simulation jitter seed")
    args = parser.parse_args()
    run(dark=args.dark, tick_interval=args.interval, seed=args.seed)


if __name__ == "__main__":
    main()
