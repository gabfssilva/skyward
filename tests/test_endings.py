"""A pool's endings: one stream of them, and the callers each waiting on a task of their own."""

import asyncio
from collections.abc import AsyncGenerator

import msgspec
import pytest

from skyward.api.v1 import TaskStateEvent
from skyward.core import endings as heard
from skyward.core.endings import Endings

pytestmark = pytest.mark.local

WAIT = 5.0


class _Stream:
    """A daemon's stream of endings that the test speaks for: each ``say`` is one frame, ``lose`` the stream giving up."""

    def __init__(self) -> None:
        self._frames: asyncio.Queue[bytes | Exception] = asyncio.Queue()
        self.asked: list[tuple[str | None, tuple[str, ...], int | None]] = []

    def say(self, task: str) -> None:
        self._frames.put_nowait(msgspec.json.encode(TaskStateEvent(compute="cmp_a", task=task, state="succeeded")))

    def lose(self) -> None:
        self._frames.put_nowait(ConnectionResetError("the daemon went away"))

    async def events(self, compute: str | None = None, *, types: tuple[str, ...] = (), after: int | None = None) -> AsyncGenerator[tuple[str, bytes]]:
        self.asked.append((compute, types, after))
        while True:
            match await self._frames.get():
                case Exception() as lost:
                    raise lost
                case bytes() as payload:
                    yield "task.succeeded", payload


async def _settled(stream: _Stream) -> tuple[Endings, asyncio.Task[None]]:
    endings = Endings(stream, "cmp_a")
    following = asyncio.create_task(endings.follow(after=41))
    await asyncio.sleep(0)
    return endings, following


def describe_a_caller_waiting_on_its_task() -> None:
    async def it_follows_the_compute_s_endings_from_where_it_was_told(stream: _Stream) -> None:
        _, following = await _settled(stream)
        following.cancel()

        assert stream.asked == [("cmp_a", heard.ENDINGS, 41)]

    async def it_returns_when_its_task_ends_and_not_before(stream: _Stream) -> None:
        endings, following = await _settled(stream)
        async with endings.watching("tsk_a") as ended:
            waiting = asyncio.create_task(ended())
            stream.say("tsk_b")
            await asyncio.sleep(0.05)
            assert not waiting.done(), "another task's ending is not this one's"

            stream.say("tsk_a")
            await asyncio.wait_for(waiting, WAIT)
        following.cancel()

    async def it_returns_at_once_for_an_ending_heard_before_it_started_waiting(stream: _Stream) -> None:
        """A task can end before the answer to its submission is back."""
        endings, following = await _settled(stream)
        stream.say("tsk_a")
        await asyncio.sleep(0.05)

        async with endings.watching("tsk_a") as ended:
            await asyncio.wait_for(ended(), WAIT)
        following.cancel()

    async def it_waits_again_for_the_next_ending_after_one(stream: _Stream) -> None:
        """A broadcast ends once per node: an ending taken leaves the next one to wait for."""
        endings, following = await _settled(stream)
        async with endings.watching("tsk_a") as ended:
            stream.say("tsk_a")
            await asyncio.wait_for(ended(), WAIT)

            second = asyncio.create_task(ended())
            await asyncio.sleep(0.05)
            assert not second.done()
            stream.say("tsk_a")
            await asyncio.wait_for(second, WAIT)
        following.cancel()

    async def it_forgets_an_ending_nobody_claimed_in_time(stream: _Stream, monkeypatch: pytest.MonkeyPatch) -> None:
        """What nobody claims is some other process's task on the same compute."""
        monkeypatch.setattr(heard, "KEPT", 0.05)
        endings, following = await _settled(stream)
        stream.say("tsk_theirs")
        await asyncio.sleep(0.1)
        stream.say("tsk_later")
        await asyncio.sleep(0.05)

        async with endings.watching("tsk_theirs") as ended:
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(ended(), 0.1)
        following.cancel()


def describe_a_stream_lost_for_good() -> None:
    async def it_fails_whoever_is_waiting_and_whoever_comes_after(stream: _Stream) -> None:
        endings, following = await _settled(stream)
        async with endings.watching("tsk_a") as ended:
            waiting = asyncio.create_task(ended())
            await asyncio.sleep(0)
            stream.lose()

            with pytest.raises(ConnectionError) as waited:
                await asyncio.wait_for(waiting, WAIT)
        with pytest.raises(ConnectionResetError):
            await following

        async with endings.watching("tsk_b") as ended:
            with pytest.raises(ConnectionError):
                await asyncio.wait_for(ended(), WAIT)
        assert isinstance(waited.value.__cause__, ConnectionResetError)


@pytest.fixture
def stream() -> _Stream:
    return _Stream()
