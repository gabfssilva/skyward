"""The parts of the document Litestar cannot derive from the handlers.

Four things are missing from a schema built purely by inference, and each is
missing for its own reason.

A **tag** has no code to be inferred from — it is the name of a resource family,
and what a compute *is* lives in prose or nowhere. A **component schema** could
be inferred and is not: Litestar reads a ``msgspec.Struct``'s fields and ignores
its docstring, so the paragraph sitting above the type never reaches the reader.
A **discriminator** cannot be inferred because the tag that discriminates is a
literal on eight separate structs and nothing says they form a union. And a
**raw request body** is invisible: the two upload routes read ``request.body()``
rather than declaring a parameter, because a ``bytes`` parameter makes Litestar
try to decode the body as JSON.

All of it is applied to the schema object after the app is built, which is what
puts it on the daemon's own ``/v1/schema`` as well as in the published document.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from litestar import Litestar
from litestar.openapi.spec import (
    Components,
    Discriminator,
    OpenAPIFormat,
    OpenAPIMediaType,
    OpenAPIResponse,
    OpenAPIType,
    Operation,
    PathItem,
    Reference,
    RequestBody,
    Schema,
    Tag,
)
from msgspec import Struct

from skyward.server.application import ssh
from skyward.shared import schemas

BLOB = "application/vnd.skyward.blob"

TAGS: tuple[Tag, ...] = (
    Tag(
        name="computes",
        description=(
            "A compute is a set of machines held under one intention: the `spec` you asked for, and the `status` the "
            "reconciler observed. It outlives the process that created it — the SDK's `Compute` is a client holding a "
            "lease on this row, not its owner.\n\n"
            "Resizing is a `PATCH` of the `spec`. Replacing anything else about it is a new **generation**, which keeps "
            "the same id and rebuilds the machines under it. The gap between `generation` and "
            "`status.observed_generation` is the progress bar, and it is why there is no operation resource to poll."
        ),
    ),
    Tag(
        name="nodes",
        description=(
            "A node is one machine, ranked, as the control plane knows it. Nodes are not created directly: they exist "
            "because a compute's `spec.nodes` asked for them, and the reconciler wrote the rows.\n\n"
            "A row appears in `requested` **before** the provider is asked for a machine, which is what makes "
            "provisioning idempotent — a machine being bought right now already counts, so the next pass does not buy a "
            "second one. A node that died stays listed until the provider confirms the instance is gone."
        ),
    ),
    Tag(
        name="tasks",
        description=(
            "A task is one call — `function` + `args` → `result` — and it is what an SDK `Future[T]` points at. It is "
            "append-only and has exactly one terminal outcome.\n\n"
            "An **execution** is a physical attempt at it. A retry creates another execution, never another task, which "
            "is what lets a handle survive the node that died under it. The task's `state` is derived from its "
            "executions and never written beside them."
        ),
    ),
    Tag(
        name="functions",
        description=(
            "A function is a registered piece of code, named by the hash of its serialized bytes. Registering is "
            "content-addressed and therefore idempotent: the same code uploaded twice is one row, however many tasks "
            "call it.\n\n"
            "This is the metadata — size, codec, the name it was registered under. The code itself is a blob."
        ),
    ),
    Tag(
        name="blobs",
        description=(
            "Content addressed by its SHA-256. Code, arguments and results all travel this way rather than inside a "
            "task body: the same argument broadcast to a hundred nodes is stored once, and a `PUT` of content already "
            "here writes nothing.\n\n"
            "Reading does not consume. A result read twice is the same bytes twice."
        ),
    ),
    Tag(
        name="providers",
        description=(
            "A provider is a named **account**, not a kind of cloud: two AWS accounts, two Vast keys and two regions "
            "coexist, and a compute names the one it wants.\n\n"
            "Credentials are validated against the kind's declared fields before the row is written, are stored here "
            "and nowhere else, and are returned by no read path. `GET /provider-kinds` is the capability negotiation "
            "that happens before any of this: a kind absent from that list cannot be registered."
        ),
    ),
    Tag(
        name="offers",
        description=(
            "The cached hardware catalog — what each registered account currently sells, normalized into one vocabulary "
            "so an `h100` from one provider compares against an `h100` from another.\n\n"
            "A cache, not a ledger: a refresh replaces a provider's rows wholesale, because an offer that vanished "
            "upstream must vanish here. A refresh that *fails* leaves the stale rows in place, since stale offers beat "
            "no offers."
        ),
    ),
    Tag(
        name="events",
        description=(
            "One log per compute, and the only place anything observable is written: node output, bootstrap phases, "
            "lifecycle transitions and task outcomes all go here. There is no `logs` resource holding a second copy.\n\n"
            "It is a stream rather than a status to poll, and it replays from any cursor that was ever valid — nothing "
            "garbage-collects it, which is what lets a client reconnect and print a bootstrap it was not around to "
            "watch."
        ),
    ),
    Tag(
        name="health",
        description=(
            "Whether the daemon works, and whether what it leans on does. Liveness is about the process; readiness is "
            "about being able to serve; dependencies are informational.\n\n"
            "A provider being down degrades the computes on it and never fails liveness — killing a daemon over one "
            "marketplace having an afternoon would take every other compute with it."
        ),
    ),
    Tag(
        name="forward",
        description=(
            "A TCP port on a node, reachable from the caller. Each connection is two requests paired by a `cid` the "
            "caller mints: a streaming `POST .../up` carrying bytes in, and a `GET .../down` carrying bytes back.\n\n"
            "It takes two because HTTP/1.1 will not carry both directions of a live socket on one request. Neither half "
            "is resumable — a dropped stream is a dropped connection."
        ),
    ),
    Tag(
        name="shell",
        description=(
            "A pseudo-terminal on a node, in the same two-request shape as forwarding: keystrokes up, paint down, "
            "paired by `cid`.\n\n"
            "This reaches the *machine*, not the worker, so it answers while the worker is busy running a training "
            "loop. Running code as a task is the other thing, and `POST /tasks` is where it goes."
        ),
    ),
    Tag(
        name="files",
        description=(
            "Reading, writing and listing paths on the machines, and running a shell command across them. Every route "
            "takes a `node` — a rank, or `all`.\n\n"
            "Writes default to every node, because data a task will read has to be wherever the task lands, and which "
            "node that is belongs to the dispatcher. Reads default to one, because four machines hold four files and "
            "concatenating them would answer a question nobody asked."
        ),
    ),
)
"""What each family of routes is, before the routes themselves."""

EVENTS = "/v1/events"

UPLOADS: tuple[tuple[str, str], ...] = (
    ("/v1/blobs/{sha256}", "Arguments, code, or a result — whatever the hash names."),
    ("/v1/functions/{sha256}", "The serialized function: cloudpickle, then compressed."),
)


def describe(app: Litestar) -> None:
    """Fill in what inference left blank, in place, on the schema already built.

    Every lookup here names a path or a component that the controllers above
    define, so a miss is a rename that has already broken the document. It raises
    rather than skipping: a spec that quietly stops describing its own event
    stream is worse than one that fails to build.
    """
    _document_schemas(app)
    _discriminate(app)
    _declare_bodies(app)


def _document_schemas(app: Litestar) -> None:
    """A struct's docstring becomes its component's description.

    Litestar reads a ``msgspec.Struct`` for its fields and never for its
    docstring, so without this the paragraph explaining what a ``Compute`` is
    stays in the source and the reader of the document gets a list of fields.

    A generic's component is named after the alias Litestar built for it —
    ``Page_skyward.shared.schemas.Task_`` — so the lookup is on what comes before
    the parameter. ``ssh`` is read alongside the wire types because one of its
    structs is served: what a command said on a machine.
    """
    structs = {
        name: value.__doc__.strip()
        for module in (schemas, ssh)
        for name, value in vars(module).items()
        if isinstance(value, type) and issubclass(value, Struct) and value.__doc__
    }

    for name, schema in _components(app).items():
        if doc := structs.get(name.split("_", 1)[0] if name.startswith("Page_") else name):
            schema.description = doc


def _discriminate(app: Litestar) -> None:
    """Point the event union's ``oneOf`` at the tag that tells its members apart.

    msgspec writes the tag as a ``const`` on each member, which is enough to
    decode and not enough to navigate: without this a reader has to open all eight
    schemas to learn which one a ``node.console`` frame is.
    """
    components = _components(app)
    for media in _bodies(_operation(app, EVENTS, "get").responses or {}, "200"):
        union = _inline(media.schema)
        members = [_named(alternative) for alternative in union.one_of or ()]
        union.discriminator = Discriminator(
            property_name="type",
            mapping={_tag(components, name): f"#/components/schemas/{name}" for name in members},
        )


def _declare_bodies(app: Litestar) -> None:
    """Say that the two upload routes take raw bytes.

    Neither declares a body parameter: both read ``request.body()``, because a
    handler annotated ``data: bytes`` makes Litestar decode the body as JSON and
    refuse the first byte that is not one. The route is right and the schema is
    silent, so the schema is written here.
    """
    for path, description in UPLOADS:
        _operation(app, path, "put").request_body = RequestBody(
            content={
                BLOB: OpenAPIMediaType(
                    schema=Schema(
                        type=OpenAPIType.STRING,
                        format=OpenAPIFormat.BINARY,
                        description="The bytes themselves — what they hash to is the name they are stored under.",
                    )
                )
            },
            description=description,
            required=True,
        )


def _components(app: Litestar) -> dict[str, Schema]:
    match app.openapi_schema.components:
        case Components(schemas=dict() as schemas):
            return schemas
        case _:
            raise LookupError("the document has no component schemas")


def _operation(app: Litestar, path: str, method: str) -> Operation:
    match (app.openapi_schema.paths or {}).get(path):
        case PathItem() as item if isinstance(operation := getattr(item, method), Operation):
            return operation
        case _:
            raise LookupError(f"no {method.upper()} {path} to describe")


def _bodies(responses: Mapping[str, OpenAPIResponse | Reference], status: str) -> Iterator[OpenAPIMediaType]:
    match responses.get(status):
        case OpenAPIResponse(content=dict() as content):
            yield from content.values()
        case _:
            raise LookupError(f"no {status} body to describe")


def _inline(schema: Schema | Reference | None) -> Schema:
    """The schema itself, for the one place that is known to carry it inline."""
    if not isinstance(schema, Schema):
        raise TypeError(f"expected an inline schema, found {schema!r}")
    return schema


def _named(alternative: Schema | Reference) -> str:
    if not isinstance(alternative, Reference):
        raise TypeError(f"expected a reference to a component, found {alternative!r}")
    return alternative.ref.rsplit("/", 1)[-1]


def _tag(components: Mapping[str, Schema], name: str) -> str:
    """The literal ``type`` a union member carries, which is what discriminates it."""
    match components.get(name):
        case Schema(properties={"type": Schema(const=str() as tag)}):
            return tag
        case _:
            raise LookupError(f"{name} carries no `type` tag to discriminate on")


__all__ = ["TAGS", "describe"]
