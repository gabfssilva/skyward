from __future__ import annotations

from skyward2.application.errors import RevisionConflictError


def etag(revision: int) -> dict[str, str]:
    return {"ETag": f'"{revision}"'}


def revision_of(if_match: str) -> int:
    try:
        return int(if_match.strip('"'))
    except ValueError:
        raise RevisionConflictError("If-Match is not a valid revision", if_match=if_match) from None
