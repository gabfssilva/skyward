# CLI

The `sky` command is a thin client over the daemon's HTTP API. It decides nothing: every command turns a few flags into one or two HTTP calls and prints what came back. That is why it holds no state and needs no daemon of its own — it either dials a remote one or runs an embedded one in its own process, and the command cannot tell which it got.

## Installation

The CLI library is an opt-in extra. A node installs `skyward` to run somebody's training loop and has no business acquiring an argument parser to do it:

```bash
pip install "skyward[cli]"
```

Without it, `sky` exits with `the sky CLI needs: pip install 'skyward[cli]'`. Running the daemon yourself additionally needs the `server` extra.

## Where the commands go

Every command resolves its target the same way:

1. an explicit `--url`,
2. otherwise `SKYWARD_URL`,
3. otherwise an **embedded daemon** in this process, against the database at `--database` (default `~/.skyward/skyward.sqlite`).

The third case is what makes the CLI usable with no setup at all: `sky offers list` works before you have started anything. It also means a command with no `--url` reads the same local database an embedded `Compute(...)` would.

Two commands are exceptions to the pure-client rule. `sky compute create` registers the provider account from *this* process, because the daemon never reads your environment for credentials. And `sky server` manages a local process rather than calling one.

## Output

Every command that prints rows takes `--output` (`-o` on `sky server status`):

- `table` — aligned columns, `-` for a missing value. The default.
- `json` — a JSON array of objects, keyed by column name.

There is no `rich` dependency; padded strings are enough for a table.

## `sky version`

```bash
sky version
# → skyward 2.0.0, python 3.13.1
```

## `sky server`

Runs the daemon — the Litestar app at `skyward.server.app` — or reports on one.

### `start`

Detached by default: it spawns `python -m uvicorn` in a new session, writes the pid to `~/.skyward/server.pid`, sends output to `~/.skyward/server.log`, and returns once `/v1/health/live` answers.

```bash
sky server start                    # 127.0.0.1:7590, detached
sky server start --port 8080
sky server start --host 0.0.0.0
sky server start --timeout 60       # seconds to wait for liveness
sky server start --foreground       # attached; Ctrl+C stops it
```

If the pidfile names a live process, `start` refuses and tells you to stop it first. If liveness never arrives, the spawned process is killed, the pidfile removed, and the log path reported.

`--foreground` runs uvicorn in this process and never touches the pidfile — which is what a dev loop wants, and also means `sky server stop` cannot stop it.

### `stop`

Signals the recorded pid with `SIGTERM` and waits for it to go. There is deliberately **no shutdown endpoint**: anything that could reach the API would otherwise be able to take the whole control plane down.

```bash
sky server stop
sky server stop --timeout 30
```

A pidfile pointing at a dead process is cleared and reported as such.

### `status`

```bash
sky server status
sky server status --url http://host:7590
sky server status -o json
```

Prints the resolved URL, the recorded pid if it is alive, and whether a daemon answers. With no `--url` and no `SKYWARD_URL`, it probes `--host`/`--port` (default `127.0.0.1:7590`).

## `sky compute`

Five verbs: `create`, `list`, `get`, `view`, `delete`. There is no `run`, no file transfer, and no interactive shell — those need exec and transfer surfaces the HTTP API does not have yet.

### `create`

```bash
sky compute create --provider runpod
sky compute create --provider aws --accelerator A100 --nodes 4 --region us-east-1
sky compute create --provider vastai --accelerator H100 --cpus 16 --memory 64 --name research
```

Flags: `--provider` (required, a provider *kind*: `aws`, `container`, `gcp`, `hyperstack`, `jarvislabs`, `lambda`, `massed_compute`, `novita`, `runpod`, `scaleway`, `tensordock`, `vastai`, `verda`, `vultr`), `--name`, `--accelerator`, `--nodes` (default 1), `--region`, `--cpus`, `--memory` (GB).

The credentials come from your environment through the SDK's provider factory, and the account is registered with the daemon if it does not already know it. The command returns as soon as the compute is accepted — provisioning happens on the daemon, and the row comes back in whatever state it is in.

There is no TOML file and no named-pool resolution; the spec is what the flags say.

### `list` and `get`

```bash
sky compute list
sky compute list --state ready
sky compute get research          # by id or by name
sky compute get research -o json
```

Columns: id, name, state, ready, total, generation, created.

### `view`

`get`, plus the machines the compute is standing on:

```bash
sky compute view research
```

Prints the compute row, then one row per node: id, rank, state, desired, machine, address, accelerator, price per hour.

### `delete`

```bash
sky compute delete research
```

The delete is accepted, not done. Reconciliation runs until the provider confirms the machines are gone, so the row that comes back still reads `deleting`. The call is conditional on the revision it just read, so a concurrent change is a refusal rather than a lost update.

## `sky log`

A compute's recorded event log — node output, bootstrap phases, task outcomes — replayed from the beginning and optionally followed.

```bash
sky log research                   # replay, as a table
sky log research -n 50             # last 50 events
sky log research -o json
sky log research -f                # keep streaming
sky log research --idle 3          # seconds of quiet that end a replay
```

The stream never ends on its own, so a command that is not following stops once the replay goes quiet; `--idle` is how long quiet has to last (default 1 second).

### `export`

```bash
sky log export research history.jsonl
sky log export research history.md
```

The format follows the suffix: `.jsonl`/`.json` (one JSON object per event) or `.md`/`.markdown` (console output grouped by node, everything else listed after it). Any other suffix is refused.

## `sky offers`

What a GPU costs, and where. The daemon owns the catalog and its per-provider TTL; these commands turn flags into query parameters.

```bash
sky offers list
sky offers list --accelerator H100 --min-count 4 --min-vram 80 --limit 10
sky offers list --provider runpod --max-price 2.5
sky offers list --limit 0                    # everything
sky offers list --refresh                    # refetch first

sky offers fetch                             # refetch, then count per provider
sky offers fetch --provider vastai

sky offers summary                           # cheapest/average/dearest per accelerator+provider
sky offers summary --accelerator A100
```

`list` sorts cheapest first and takes `--provider`, `--accelerator`, `--min-count`, `--min-vram`, `--max-price`, `--limit` (default 20; `0` for all) and `--refresh`. `summary` takes the same filters minus `--limit`.

`fetch` is not a separate route — it is `--refresh` on the same query, which is why it can also filter. A provider that fails to answer keeps its stale rows and records the failure on itself, so the counts are what the catalog holds, not what the fetch won.

There is no SQL query command: the catalog lives behind the daemon and there is no endpoint for raw SQL.

## `sky providers`

A provider is a registered *account*, not a kind. The daemon holds the credentials and is the only thing that ever tries them.

```bash
sky providers list                 # the registered accounts
sky providers list --kinds         # what can be registered
sky providers check                # every account
sky providers check aws-default    # one, by id or name
```

`list` shows name, kind, id, offers count, and when they were last fetched. `--kinds` instead lists every provider kind this build supports, the credential fields it needs, and how long its offers stay fresh.

`check` is a **read, not a probe**. The daemon never returns credentials, and the CLI never authenticates on its own; the row already carries the last error the daemon hit and the last successful fetch, which is the answer a probe would have gone looking for. The verdict is `ok`, `error` (with the message), or `unused` when the credentials have never been exercised.

## `sky config`

There is no configuration file. Configuration here is the resolution a command would perform, made visible.

```bash
sky config path        # database path and resolved URL
sky config show        # url, where it came from, database path, whether it exists
sky config validate    # is the resolved daemon reachable and ready?
```

`validate` calls `/v1/health/ready` and exits non-zero when it is not.

## Scripting

```bash
# wait for a compute to be ready
until [ "$(sky compute get research -o json | jq -r '.[0].state')" = "ready" ]; do sleep 2; done

# cheapest H100 provider
sky offers list --accelerator H100 --limit 1 -o json | jq -r '.[0].PROVIDER'

# follow a compute's log into a file
sky log research -f -o json >> research.jsonl
```

## Next steps

- [Getting Started](getting-started.md) — installation, credentials, first remote computation
- [Core concepts](concepts.md) — computes, operators, and the model the CLI exposes
- [Providers](providers.md) — per-provider setup and credentials
- [Events](reference/events.md) — the event stream `sky log` reads
