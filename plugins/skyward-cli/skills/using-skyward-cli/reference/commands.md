# `sky` command reference

Every command below accepts `--url` (daemon URL, overrides `SKYWARD_URL`; defaults to `http://127.0.0.1:17590`). Commands that render a table accept `--output table|json`. `sky <command> --help` is authoritative for the installed build.

`REF` is a compute id **or** name. `--node` takes `all` or a rank (`0`, `1`, …), except `download`, which takes a rank only.

---

## Top level

| Command | Does |
|---|---|
| `sky version` | Print the Skyward and Python versions |
| `sky new …` | Alias of `sky compute create` |
| `sky status [REF]` | One compute, or all of them |
| `sky sessions` | Every compute — `status` with no argument |
| `sky stop REF` | Tear a compute down (alias of `compute delete`) |
| `sky monitor REF [--mode rich\|log]` | Watch a live compute until interrupted |
| `sky console REF [--node N] [--command CMD]` | Interactive shell on one machine |
| `sky repl REF [--node N]` | Python REPL on the node's bootstrapped interpreter |

---

## `sky server` — the daemon

```
sky server start [--host 127.0.0.1] [--port 17590] [--foreground] [--timeout 30.0] [--database PATH]
sky server stop  [--timeout 10.0]
sky server status [--url URL] [--host 127.0.0.1] [--port 17590] [-o table|json]
```

- `start` detaches by default, records the pid in `~/.skyward/server.pid`, logs to `~/.skyward/server.log`, and waits up to `--timeout` for liveness before giving up.
- `~/.skyward/server.log` is the only place a failed provider launch is reported; the compute's event log does not carry it.
- `--foreground` runs attached and ends with the terminal.
- `--database` is passed to the spawned process as `SKYWARD_DATABASE`. It is the only command that takes one.
- `stop` signals the recorded pid. There is no shutdown endpoint — anything that could reach the API could otherwise take the control plane down.
- `stop` also ends a daemon the SDK started: a `sky.Compute` that names none starts one at the same address, recorded in the same pidfile.
- `start` needs `uvicorn` (`skyward[server]`).

---

## `sky config` — what a call resolved

```
sky config path      [--url URL] [--output table|json]
sky config show      [--url URL] [--output table|json]
sky config validate  [--url URL] [--output table|json]
```

- `path` — the resolved daemon URL, and the database a daemon started here would use.
- `show` — url, its source (`flag` / `environment` / `default`), that database, whether it exists.
- `validate` — GETs `/v1/health/ready`; exits 1 when the daemon is unreachable or not ready.

There is no configuration file. These print resolution, not settings.

---

## `sky providers` — the accounts the daemon provisions with

```
sky providers list [--kinds] [--output table|json]
sky providers check [NAME] [--output table|json]
sky providers set KIND [--config key=value]... [--name NAME] [--output table|json]
```

- `list` — registered accounts (name, kind, id, offers, fetched).
- `list --kinds` — what this build supports: kind, required credential fields, offers TTL.
- `check` — per account: `ok`, `error` (with the daemon's last error) or `unused` (credentials never exercised). A read of the stored row, not a live probe.
- `set` — register or update. `--config` values are validated against the account struct, so an unknown field is refused here. **Credentials are read from the environment, never from a flag.** `--name` defaults to the kind, which is what a compute created without a name looks for.

Kinds: `aws`, `container`, `gcp`, `hyperstack`, `jarvislabs`, `lambda`, `massed_compute`, `novita`, `runpod`, `scaleway`, `tensordock`, `vastai`, `verda`, `vultr`.

---

## `sky offers` — the catalog

```
sky offers list    [--provider P] [--accelerator A] [--min-count N] [--min-vram GB] [--max-price X] [--limit 20] [--refresh]
sky offers summary [--provider P] [--accelerator A] [--min-count N] [--min-vram GB] [--max-price X] [--refresh]
sky offers fetch   [--provider P]
```

- `list` — cheapest first. Columns: provider, kind, instance, accelerator, vram, cpus, memory, region, spot, on-demand, unit. `--limit 0` prints everything.
- `summary` — per (accelerator, provider): offer count, cheapest, average, dearest.
- `fetch` — force a refetch and report how many offers each provider returned. A provider that fails keeps its stale rows and records the failure on itself, so the count is what the catalog holds, not what the fetch won.
- `--refresh` is the same forced refetch, on `list` and `summary`.
- `--max-price` is in the offer's own billing unit.

An empty listing on a daemon with no registered accounts is reported as such on stderr — that is a different problem from a filter that excluded everything.

---

## `sky compute` — computes and the work on them

### Lifecycle

```
sky compute list [--state STATE]
sky compute get REF
sky compute view REF
sky compute create --provider KIND [--name N] [--accelerator A] [--nodes 1]
                   [--region R] [--cpus N] [--memory GB] [--ttl SECONDS]
sky compute scale REF --nodes N|MIN:MAX
sky compute delete REF
```

- `list`/`get` columns: id, name, state, ready, total, generation, created.
- `view` adds the node table: id, rank, state, desired, machine, address, accelerator, $/h.
- `create` returns immediately; the machines arrive afterwards. It registers the provider account from *this* process if the daemon has none, because the daemon never reads the environment.
- `--ttl` is the dead-man switch a supporting provider arms on each machine: with nobody connected for that long, the machine removes itself. `0` never does. The default is the spec's, and it is short.
- `scale --nodes N` fixes the size; `--nodes MIN:MAX` makes it elastic (`1 <= MIN <= MAX`). What comes back is a new `generation`, not a finished resize.
- `delete` is accepted, not done — the compute stays `deleting` until the provider confirms the machines are gone.
- Writes are guarded by `If-Match` on the compute's revision and retried five times on a `revision_conflict`, since the reconciler and the lease also move it.

### Work

```
sky compute exec REF COMMAND... [--node all]
sky compute run  REF SCRIPT [ARGS...] [--all]
```

- `exec` runs in the **machine's shell** — questions about the node, and it reaches a node whose worker is busy. Quote a command carrying flags, or put it after `--`: `sky compute exec training --node 0 -- df -h`.
- `run` submits a **task**: the local script travels the path a `@sky.function` takes, landing in a worker with the image, plugins and runtime API around it. Its output streams back over the compute's event log as it prints. `--all` broadcasts; the default is one node.
- Both exit with the worst node's status.

### Files

```
sky compute ls       REF PATH [--node 0]
sky compute rm       REF PATH [--node all]
sky compute upload   REF LOCAL REMOTE [--node all]
sky compute download REF REMOTE LOCAL [--node 0]
```

- `upload` writes to every node by default: a file a task will read has to be wherever the task lands, and which node that is belongs to the dispatcher.
- `download` reads from one node, by rank — four machines hold four files.
- `rm` is recursive.

---

## `sky log` — the compute's event log

```
sky log COMPUTE [-f|--follow] [-n|--limit N] [--idle 1.0] [--output table|json]
sky log export COMPUTE FILE [-n|--limit N] [--idle 1.0]
```

The daemon keeps one log per compute and serves it over SSE, replayed from the beginning and then followed — node output, phase marks and task outcomes are all events in it, so attaching late loses nothing.

- Without `--follow`, the replay ends after `--idle` seconds of quiet.
- `export` writes `.jsonl` (one JSON object per event) or `.md` (console output grouped per node, everything else listed after). Any other suffix is refused.
- Event names include `node.console` (a printed line) and `node.phase` (bootstrap progress).

---

## `sky notebook` — the Jupyter kernel

```
sky notebook install COMPUTE [--url URL] [--directory DIR]
sky notebook remove  COMPUTE [--directory DIR]
```

Writes (or deletes) one kernelspec recording which compute to attach to and through which daemon. Then pick `Skyward (<compute>)` in Jupyter. Needs `skyward[notebook]`.
