from collections.abc import Callable
from functools import wraps


def rethrow[**P, R, E: BaseException, NewE: BaseException](
    catch: type[E],
    into: Callable[[E], NewE],
) -> Callable[[Callable[P, R]], Callable[P, R]] | Callable[P, R]:
    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                return fn(*args, **kwargs)
            except catch as e:
                raise into(e) from e

        return wrapper

    return decorator
