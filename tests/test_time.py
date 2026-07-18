from skyward import time as time


def test_minutes():
    assert time.minutes(30) == 1800.0
    assert time.minutes(0) == 0.0
    assert time.minutes(1.5) == 90.0


def test_hours():
    assert time.hours(1) == 3600.0
    assert time.hours(0) == 0.0
    assert time.hours(2.5) == 9000.0


def test_days():
    assert time.days(7) == 604800.0
    assert time.days(0) == 0.0
    assert time.days(1) == 86400.0


def test_returns_float():
    assert isinstance(time.minutes(1), float)
    assert isinstance(time.hours(1), float)
    assert isinstance(time.days(1), float)


def test_all():
    assert time.__all__ == ["minutes", "hours", "days"]
