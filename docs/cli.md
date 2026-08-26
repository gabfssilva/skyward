# CLI

`sky` is the command-line client for the Skyward control plane. It can talk to a remote daemon or use an embedded daemon in the current process.

## Installation

Install the CLI separately from the SDK:

```bash
pip install "skyward[cli]"
```

To run a local daemon from the CLI, install both extras:

```bash
pip install "skyward[cli,server]"
```

The other optional extras are `tui` for the terminal UI, `notebook` for the Jupyter provisioner, `storage` for S3-compatible storage, and `client` for remote HTTP access from the SDK. Provider-specific extras are not required.

## Daemon resolution

Commands resolve the daemon in this order:

1. `--url`;
2. `SKYWARD_URL`;
3. an embedded daemon using `--database`, or `~/.skyward/skyward.sqlite` by default.

When a URL resolves, `--database` is ignored. `sky config` shows the effective resolution.

Commands that render rows accept `--output table` or `--output json`. The default is `table`; use JSON for scripts.

## `sky version`

```bash
sky version
```

## `sky server`

`sky server` manages a local daemon process. The detached process writes its PID to `~/.skyward/server.pid` and its output to `~/.skyward/server.log`.

```bash
sky server start
sky server start --host 0.0.0.0 --port 8080
sky server start --foreground
sky server stop
sky server status
sky server status --url http://host:7590
```

`start` waits for `/v1/health/live`. `--foreground` keeps the daemon attached to the terminal and does not create a PID file. `stop` only stops a process started by this CLI.

## `sky compute`

### Create

`create` registers the provider account from the current process and submits a compute. It returns without waiting for the compute to become ready.

```bash
sky compute create --provider aws
sky compute create --provider aws --accelerator A100 --nodes 4 --region us-east-1
sky compute create --provider runpod --accelerator RTX_4090 --name research
```

The supported provider kinds are:

`aws`, `container`, `gcp`, `hyperstack`, `jarvislabs`, `lambda`, `massed_compute`, `novita`, `runpod`, `salad`, `scaleway`, `tensordock`, `vastai`, `verda`, and `vultr`.

A kind whose SDK extra is not installed is not registered, so it will not appear. `sky providers list --kinds` shows what this installation can actually reach.

The available create flags are `--provider`, `--name`, `--accelerator`, `--nodes`, `--region`, `--cpus`, `--memory`, `--url`, `--database`, and `--output`. The provider account reads credentials from the current process. Credential values are not printed by the CLI.

`sky new` is an alias for `sky compute create`:

```bash
sky new --provider container --name local
```

### Scale

`scale` changes how many machines a compute stands on, without replacing the ones already up:

```bash
sky compute scale research --nodes 8
sky compute scale research --nodes 2:8
```

`--nodes N` is a fixed size; `--nodes MIN:MAX` is an elastic range, the same thing `nodes=(2, 8)` means in the SDK. The bounds are written whole, so the flag is the compute's new size and not a patch on the old one.

Like `create`, it returns without waiting: what comes back is a new `generation`, and the machines are bought or drained by reconciliation afterwards. A compute running a collective plugin (`torch`, `jax`, `accelerate`) is refused — its process group was formed with the ranks it started with, and one added now would block in it.

### Read and delete

```bash
sky compute list
sky compute list --state ready
sky compute get research
sky compute get research --output json
sky compute view research
sky compute delete research
```

`get` and `view` accept a compute id or name. `view` also prints the node rows. `delete` is an intent change: the returned state can remain `deleting` while the daemon reconciles the provider state.

### Files and commands

These commands use the daemon's file and execution endpoints:

```bash
sky compute ls research /data
sky compute rm research /tmp/output
sky compute upload research ./data.csv /data/data.csv
sky compute download research /data/result.json ./result.json
sky compute exec research --node 0 nvidia-smi
sky compute run research train.py
sky compute run research --all train.py
```

`ls`, `rm`, and `upload` target every node by default where the command allows it. `download` reads one node and defaults to rank `0`. `exec` runs a shell command on the selected nodes. `run` sends a local Python script through the worker path; `--all` runs it on every node.

## `sky log`

`sky log` replays a compute's recorded events. Without `--follow`, it stops after the replay is quiet.

```bash
sky log research
sky log research --follow
sky log research --limit 50 --output json
sky log research --idle 3
sky log export research history.jsonl
sky log export research history.md
```

`export` accepts `.jsonl`, `.json`, `.md`, and `.markdown` destinations.

## `sky offers`

Offers are served from the daemon's provider cache. `list` sorts by the cheapest available price and supports:

```bash
sky offers list
sky offers list --accelerator H100 --min-count 4 --min-vram 80 --limit 10
sky offers list --provider runpod --max-price 2.5
sky offers list --refresh --output json
```

The filters are `--provider`, `--accelerator`, `--min-count`, `--min-vram`, `--max-price`, `--limit`, and `--refresh`. `--provider` accepts an account id or name. `--accelerator` accepts the provider's spelling; returned offers use the shared normalized accelerator vocabulary. `--limit 0` prints all rows.

`fetch` forces a refresh and reports the number of cached rows per provider:

```bash
sky offers fetch
sky offers fetch --provider vastai
```

`summary` groups cached rows by accelerator and provider:

```bash
sky offers summary
sky offers summary --accelerator A100 --refresh
```

Each provider has its own freshness interval. If a refresh fails, the daemon keeps the provider's stale rows and records the error on the provider account.

## `sky providers`

A provider is a registered account, not only a provider kind. The daemon uses the account credentials; list and check responses do not return them.

```bash
sky providers list
sky providers list --kinds
sky providers set runpod --config cloud_type=community
sky providers set aws --name production --config region=eu-west-1
sky providers check
sky providers check production
```

`list` shows registered accounts. `list --kinds` shows supported kinds, required credential fields, and offer-cache TTLs. `check` reports the last recorded result; it does not perform a new credential probe.

`set` registers an account, or rewrites the settings of one already registered. `--config` takes the account's own fields as `key=value`, repeatable, and they are validated against the account before anything is sent. Credentials are never written here: they are read from the environment, the same way a pool reads them. The row is what the daemon builds its adapter from, so a compute created with `--provider runpod` provisions with whatever `set` last wrote.

## `sky config`

Skyward has no configuration file. These commands show the resolved daemon URL and embedded database:

```bash
sky config path
sky config show
sky config validate
```

`validate` checks `/v1/health/ready` and exits non-zero when the daemon is not reachable or not ready.

## Top-level compute aliases

The following commands are shortcuts for common compute operations:

```bash
sky status
sky status research
sky sessions
sky stop research
```

`status` lists all computes when no reference is supplied and reads one compute otherwise. `sessions` lists all computes. `stop` delegates to compute deletion.

## Interactive commands

```bash
sky console research
sky console research --node 0
sky console research --node 0 --command "nvidia-smi"
sky repl research
```

`console` opens a shell on one node. `repl` opens the Python interpreter bootstrapped on that node.

`sky monitor` attaches to a running compute and follows it until you interrupt:

```bash
sky monitor research
sky monitor research --mode log
```

`--mode rich` (the default) draws the live footer; `--mode log` prints plain lines, which is what you want in CI or when piping to a file. Monitoring creates nothing — the compute has to exist already.

## Next steps

- **[Getting started](getting-started.md)** — installation and credential setup
- **[Providers](providers.md)** — the accounts `sky providers` lists
- **[Events](reference/events.md)** — the stream `sky log export` reads
- **[Notebook kernels](notebook.md)** — running a Jupyter kernel on a compute
