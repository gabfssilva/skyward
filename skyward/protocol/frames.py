"""What a worker says about a task, in the words all three processes know.

The node produces these, the daemon reads them, the SDK unwraps them. They are
msgpack — the payloads inside are pickled and are opened only at the two ends that
have the user's libraries, never in the middle.
"""

from __future__ import annotations

from msgspec import Struct


class Done(Struct, frozen=True, tag="done", tag_field="status"):
    value: bytes


class Failed(Struct, frozen=True, tag="failed", tag_field="status"):
    error: str
    traceback: str


class Chunk(Struct, frozen=True, tag="chunk", tag_field="status"):
    """One thing a generator yielded."""

    value: bytes


class End(Struct, frozen=True, tag="end", tag_field="status"):
    """The generator is exhausted. Said once, to the daemon, and never forwarded."""


class Pending(Struct, frozen=True, tag="pending", tag_field="status"):
    """The worker has the task and has not finished it."""


class Unknown(Struct, frozen=True, tag="unknown", tag_field="status"):
    """The worker has never heard of the task."""


type Outcome = Done | Failed
type Lookup = Done | Failed | Pending | Unknown

type Step = Chunk | Failed | End
"""What one pull on a generator produces, as the worker answers the daemon."""

type Frame = Chunk | Failed
"""What a streaming task sends back: items, and at most one failure, which ends it.

A failure travels as a frame and not as an exception because half of it is already
on the caller's side — the frames it consumed are consumed. There is no status code
left to fail with, so the failure is the last thing the stream says.

``End`` is not among them: the caller's iterator ends when the body does, and a
frame saying so would be a second way to say the same thing.
"""
