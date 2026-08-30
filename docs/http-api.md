# HTTP API

The daemon is the whole control plane, and HTTP is its only interface. The Python SDK and the `sky` CLI are clients of this API and get no privileged path into it — anything they do, you can do with `httpie`.

The surface is small on purpose: 31 paths under `/v1`. Most of what you need to know is not the routes but the four conventions that apply across all of them.

## Two families of resource

The split is what makes the rest of the API predictable.

**Declarative resources** — `compute` and `node` — carry `spec` (what you asked for) and `status` (what has been observed). `PATCH` only ever touches `spec`; `status` is written by the reconciler and never by a client. There is deliberately no `operation` or `job` resource to poll: progress is `generation` against `status.observed_generation`.

**Imperative resources** — `task` — are append-only facts with exactly one terminal outcome. Its `executions` are the physical attempts. A retry creates another execution, never another task, which is what lets an SDK `Future` keep a stable handle across a node that died under it.

```
compute ──< generation        one frozen definition per revision of the spec
        └─< node              one machine, ranked

task    ──< execution         one physical attempt, ordinal-counted

provider ─< offer             a cached hardware catalog
function ── blob              code, arguments and results, addressed by hash
```

## Optimistic concurrency

Every declarative resource carries a `revision`, served as an `ETag`:

```console
$ http GET :17590/v1/computes/cmp_7f3a1c
ETag: "7"
```

A write that changes one must say which revision it expected:

```console
$ http PATCH :17590/v1/computes/cmp_7f3a1c If-Match:'"7"' nodes:='{"initial": 8}'
```

`nodes` is the only field this accepts, and a compute running a collective plugin is refused with `422 compute_not_resizable`: its process group was formed with the ranks it started with.

If the stored revision has moved on, the write is refused with `412` and `revision_conflict`. That error is retryable: re-read, re-apply, re-send. Every successful write bumps the revision.

This is what keeps two clients — your script and a `sky compute delete` in another terminal — from silently overwriting each other's intent.

## Idempotency

Any request that creates something takes an `Idempotency-Key`:

```console
$ http POST :17590/v1/computes Idempotency-Key:k-1 spec:=@spec.json
```

The daemon stores the key alongside a fingerprint of the request. The same key with the same request is a retry, and returns the original resource rather than creating a second. The same key with a *different* request is a bug on the caller's side, and gets `409 idempotency_conflict` instead of a resource nobody asked for.

This matters more than it looks. The client cannot tell a lost response from a lost request, so without it every network blip during provisioning risks paying for a second cluster.

## Errors

Every failure is the same JSON object, whatever produced it:

```json
{
  "code": "revision_conflict",
  "message": "compute cmp_7f3a1c is at revision 9",
  "retryable": true,
  "request_id": "...",
  "details": {"if_match": "7"}
}
```

`code` is a closed set, so a client matches on it rather than parsing prose. `retryable` says whether trying again can plausibly work — it is a property of the error, not a guess for the caller to make. Which of these a given route can answer with is declared on the route, so a generated client knows before it calls.

| Code | Status | Retryable | Meaning |
|---|---:|---|---|
| `not_found` | 404 | no | No such resource |
| `revision_conflict` | 412 | yes | `If-Match` did not match the stored revision |
| `idempotency_conflict` | 409 | no | Key reused with a different request |
| `lease_held` | 409 | yes | Another process owns this compute |
| `compute_not_accepting` | 422 | no | The compute is deleting or failed |
| `compute_not_resizable` | 422 | no | The compute runs a collective, and its ranks are frozen |
| `unsupported_provider` | 422 | no | No adapter registered for that kind |
| `unsupported_plugin` | 422 | no | No plugin registered under that kind |
| `secret_in_definition` | 422 | no | A credential was put where the API serves it back |
| `hash_mismatch` | 400 | no | Uploaded content does not hash to its name |
| `task_failed` | 409 | no | Reading the result of a task that failed |
| `task_indeterminate` | 409 | no | Contact was lost after code may have run |
| `duplication_not_acknowledged` | 409 | no | Retrying an indeterminate task without accepting it may run twice |
| `capability_mismatch` | 422 | no | The provider cannot do what the spec asks — volumes, clustering |

`secret_in_definition` is worth its own note: credentials belong on a provider row, which no read path selects and the API never returns. Putting one in a compute spec is refused rather than stored, because a spec *is* served back.

## Content-addressed payloads

Code, arguments, and results never travel in a task body. They are uploaded as blobs named by their SHA-256 and referenced by that name:

```console
$ http PUT :17590/v1/blobs/<sha256> < payload.bin
$ http PUT :17590/v1/functions/<sha256> codec=cloudpickle-lz4
$ http POST :17590/v1/tasks function=<sha256> args_sha256=<sha256> compute=cmp_7f3a1c
```

The same argument broadcast to a hundred nodes is stored once. A `PUT` of content already present is a no-op, so a client can skip the upload entirely by trying the task first. Results come back the same way, which is why reading one twice does not consume it.

## Watching instead of polling

There is no status endpoint to poll on a timer. `GET /v1/events` is a Server-Sent Events stream carrying every lifecycle transition, node output, and task outcome, with replay from `Last-Event-ID`. That is how the SDK learns a compute is ready and how `sky log` prints a bootstrap it was not around to watch.

Each message's `data:` is one of eight payload types, discriminated by a `type` field inside the payload:

```json
{"type": "node.console", "compute": "cmp_7f3a1c", "node": "nod_2b91", "content": "epoch 3 loss 0.214"}
```

The frame's `event:` name is finer than that tag — ten node states share `node.state`, four task outcomes share `task.state` — and it is what the `types` filter matches on. The tag is what lets a payload be decoded on its own, once it has been written down, exported, or replayed out of the stream.

See [Events](reference/events.md) for the stream's filters, replay semantics, and behaviour with slow consumers.

## Health

```console
$ http GET :17590/v1/health/live          # the process is up
$ http GET :17590/v1/health/ready         # it can serve
$ http GET :17590/v1/health/dependencies  # what it depends on, and their state
```

`sky server start` waits on `live`; `sky config validate` checks `ready`.

## The full specification

Every route, parameter, request body, response shape, and schema is in the OpenAPI document, browsable in full:

<p style="margin: 1.2em 0;">
  <a class="md-button md-button--primary" href="../api/">Open the API explorer &rarr;</a>
</p>

It is generated from the running application by `scripts/gen_openapi.py` (`task docs:openapi`), so it cannot drift from the controllers. The raw document is at [`openapi.json`](openapi.json) if you would rather point a code generator at it.

## Next steps

- **[API explorer](../api/)** — the full OpenAPI document, rendered
- **[Persistence](persistence.md)** — what the daemon writes down, and what survives a restart
- **[Architecture](architecture.md)** — the components behind these routes
- **[Events](reference/events.md)** — the SSE stream and its replay semantics
- **[CLI](cli.md)** — the command-line client for this API
