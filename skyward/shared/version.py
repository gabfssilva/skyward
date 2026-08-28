"""Which skyward this is.

Behind a function because ``skyward/_version.py`` is written at build time from
the git tags: it is not in the tree, so a module-scope import of it would be an
import of something that may not be there yet.
"""


def current() -> str:
    """The version of the skyward that is running."""
    from skyward._version import __version__

    return __version__


__all__ = ["current"]
