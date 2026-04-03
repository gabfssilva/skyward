"""Human-readable time durations as seconds.

Usage::

    import skyward as sky

    cache = sky.time.hours(1)    # 3600.0
    ttl = sky.time.minutes(30)   # 1800.0
    expiry = sky.time.days(7)    # 604800.0
"""


def minutes(n: float, /) -> float:
    """Convert minutes to seconds."""
    return n * 60.0


def hours(n: float, /) -> float:
    """Convert hours to seconds."""
    return n * 3600.0


def days(n: float, /) -> float:
    """Convert days to seconds."""
    return n * 86400.0


__all__ = ["minutes", "hours", "days"]
