from contextlib import contextmanager, redirect_stderr, redirect_stdout
from io import StringIO
from typing import Generator, TextIO, Callable


class CallbackWriter(TextIO):
    def __init__(self, callback: Callable[[str], None]) -> None:
        self._callback = callback
        self._buffer = StringIO()

    def write(self, s: str) -> int:
        self._callback(s)
        return self._buffer.write(s)

    def getvalue(self) -> str:
        return self._buffer.getvalue()

    # Required TextIO methods
    def read(self, n: int = -1) -> str:
        return self._buffer.read(n)

    def readline(self, limit: int = -1) -> str:
        return self._buffer.readline(limit)

    def flush(self) -> None:
        self._buffer.flush()

    def close(self) -> None:
        self._buffer.close()

    def seekable(self) -> bool:
        return False

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return True


@contextmanager
def redirect_output(
    callback: Callable[[str], None],
) -> Generator[tuple[CallbackWriter, CallbackWriter], None, None]:
    out = CallbackWriter(callback)
    err = CallbackWriter(callback)
    with redirect_stdout(out), redirect_stderr(err):  # type: ignore[type-var]
        yield out, err