---
name: using-skyward-cli
description: Use the `sky` command line to provision and drive Skyward GPU computes — start the daemon, register provider accounts, browse offers and prices, create/scale/delete computes, run scripts and shell commands on nodes, move files, open a console or REPL, and read a compute's event log. Triggers on sky CLI, skyward CLI, sky compute, sky offers, sky providers, sky server, sky log, sky console, provision GPU, GPU pricing, spin up nodes.
---

# Using the Skyward CLI

`sky` is a thin client over the Skyward daemon's HTTP API. It decides nothing and
stores nothing: every command turns a few flags into one or two HTTP calls and
prints what came back. Everything that matters — computes, nodes, offers,
provider accounts, event logs — lives in the daemon.

## Where a command lands

Every command talks to a daemon — the CLI never hosts one. Resolved per call:

1. `--url http://host:port`
2. `SKYWARD_URL`
3. `http://127.0.0.1:17590`, where `sky server start` binds

So a local session is one `sky server start`, then everything else plain:

```bash
sky server start          # detaches, binds 127.0.0.1:17590, pid in ~/.skyward/server.pid
sky compute list          # no flag, no environment variable
sky config show           # url, where it came from, and the daemon's default database
sky config validate       # is it reachable and ready?
sky server stop           # SIGTERM the pid this machine recorded
```

A command that reaches nothing says so and stops:

```
$ sky compute list
no daemon at http://127.0.0.1:17590 — run: sky server start
```

`sky server start --foreground` stays attached, for a dev loop. `--database`
belongs to `sky server start` alone — no other command takes one, because no
other command owns state.

Daemon logs go to `~/.skyward/server.log`. `sky server start --database PATH`
points it at a different SQLite file (it passes `SKYWARD_DATABASE` to the
process). `sky server stop` signals the recorded pid — there is no shutdown
endpoint, deliberately.

## Provider accounts

A provider is a *registered account*, not a kind. The daemon holds it and is the
only thing that ever uses it.

```bash
sky providers list --kinds        # what can be registered, and which credentials each needs
sky providers set runpod          # register the account
sky providers set runpod --config cloud_type=community --config region=EU
sky providers set aws --config region=eu-west-1 --name aws-eu
sky providers list                # what is registered
sky providers check               # did the daemon's last use of each account work?
```

**Credentials are never passed on the command line and cannot be.** `sky
providers set` reads them from the environment the same way the SDK does, so a
key never lands in shell history. Export them first:

| kind | environment |
|---|---|
| `aws` | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` (or `~/.aws/credentials`, static keys only) |
| `gcp` | `GOOGLE_APPLICATION_CREDENTIALS` (a file path), `GOOGLE_CLOUD_PROJECT` |
| `runpod` | `RUNPOD_API_KEY` |
| `vastai` | `VAST_API_KEY` |
| `hyperstack` | `HYPERSTACK_API_KEY` |
| `lambda` | `LAMBDA_API_KEY` |
| `novita` | `NOVITA_API_KEY` |
| `tensordock` | `TENSORDOCK_API_KEY`, `TENSORDOCK_API_TOKEN` |
| `scaleway` | `SCW_SECRET_KEY`, `SCW_DEFAULT_PROJECT_ID` |
| `verda` | `VERDA_CLIENT_ID`, `VERDA_CLIENT_SECRET` |
| `vultr` | `VULTR_API_KEY` |
| `jarvislabs` | `JL_API_KEY` |
| `massed_compute` | `MASSED_API_KEY` |
| `container` | none — local Docker, the one provider needing no credentials |

`--config key=value` takes the account's own fields and is validated against the
account struct before anything is sent, so a misspelled setting is refused here
rather than at provisioning time. `--name` registers under a name other than the
kind; a compute created without a name looks for the kind.

Use `container` for a local smoke test: it runs nodes as Docker containers, no
cloud and no bill.

## Offers and prices

One GET against the daemon's catalog. The daemon owns the per-provider TTL and
the refresh a stale provider triggers.

```bash
sky offers list --accelerator H100 --min-count 8 --limit 10
sky offers list --provider runpod --max-price 2.5 --limit 0     # 0 prints everything
sky offers summary --accelerator A100                           # cheapest/average/dearest per provider
sky offers fetch                                                # force a refetch, report counts
```

An empty listing on a daemon with no registered accounts is not an answer about
hardware — the CLI says so on stderr. Register an account first.

## Computes

```bash
sky compute create --provider runpod --accelerator A100 --nodes 4 --name training
sky compute list
sky compute get training
sky compute view training          # the compute plus the machines it stands on
sky compute scale training --nodes 8
sky compute scale training --nodes 2:8      # elastic range, MIN:MAX
sky compute delete training
```

`sky new` is an alias for `sky compute create`. `sky status [ref]`,
`sky sessions` and `sky stop <ref>` are the same commands under the names you
reach for when the question is "what is running".

Three things to expect:

- **create returns immediately.** It posts intent; the machines arrive afterwards.
  Watch with `sky monitor <ref>` or `sky log <ref> -f`.
- **delete is accepted, not done.** What comes back is still `deleting`, and stays
  that way until the provider confirms the machines are gone.
- **scale returns a new `generation`**, not a finished resize. What is up is kept;
  the difference is bought or drained by reconciliation.

`--ttl SECONDS` on create is the dead-man switch the providers that support one
arm on each machine: with nobody connected for that long, the machine removes
itself rather than billing for a daemon that is never coming back. `--ttl 0`
never does.

### When `provisioning` does not move

A provider that has no stock refuses the launch and the reconciler simply tries
again, so a compute can sit in `provisioning` for many minutes with nothing
wrong. **That refusal is not in the compute's event log.** `sky log` carries
`node.requested`, `node.connecting`, `node.bootstrapping`, `node.phase`,
`node.console`, `node.ready` and `compute.cost` — the provider exception only
reaches the daemon's own log:

```bash
tail -f ~/.skyward/server.log          # where a failed launch actually says why
sky log <id> -f                        # where the node's own progress is
```

A stock-out reads like `no market could place a runpod machine` wrapping the
provider's message. It is worth waiting through — retries do land.

## Running work on a compute

```bash
sky compute exec training nvidia-smi                  # every node's shell
sky compute exec training --node 0 -- df -h           # one node, by rank
sky compute exec training "nvidia-smi -L"             # quoting works too
sky compute run training train.py --all               # a local Python script, on every node
sky compute run training train.py -- --epochs 10      # args forwarded as sys.argv
```

A command carrying its own flags has to be quoted or put after `--`; otherwise
the parser reads them as `sky`'s (`-h` prints help).

`exec` runs in the **machine's shell** — the right tool for questions about the
node (what the driver reports, what is on the disk), and it reaches a node whose
worker is busy. `run` is a **task**: the script travels the same path a
`@sky.function` takes, so it lands in a worker with the image, the plugins and
the runtime API around it, and its output streams back over the compute's event
log as it prints. Both exit with the worst node's status.

Files:

```bash
sky compute ls training /workspace --node 0
sky compute upload training ./data.csv /workspace/data.csv    # every node by default
sky compute download training /workspace/model.pt ./model.pt --node 0
sky compute rm training /workspace/scratch
```

`--node` takes `all` or a rank. `download` takes only a rank — four machines hold
four files, and there is no answer to which one was meant.

## On the node

What `exec` and `run` land in:

| | |
|---|---|
| working root | `/opt/skyward` |
| interpreter | `/opt/skyward/.venv/bin/python` (skyward and the image's pip packages) |
| `uv` | `/root/.local/bin/uv` — **not** on the `exec` shell's `PATH` |
| event log the node writes | `/opt/skyward/events.jsonl` |

`exec` gets a bare non-login shell: `PATH` is the system default, so anything the
bootstrap installed under `~/.local/bin` has to be named in full. Adding packages
after the fact is therefore:

```bash
sky compute exec <ref> "/root/.local/bin/uv pip install --python /opt/skyward/.venv/bin/python torch transformers"
```

The base image is minimal — no toolchain. PyTorch's inductor/triton shells out to
a C compiler the first time it builds a CUDA kernel and dies with *"Failed to
find C compiler"* if there is none:

```bash
sky compute exec <ref> "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential"
```

Doing this from the CLI is the fallback. The first-class way to shape a node is
`sky.Image(pip=..., apt=...)` and `plugins=[sky.plugins.Torch()]` on the SDK's
`Compute` — `sky compute create` has no flag for either.

## Watching a compute

```bash
sky monitor training                # live Rich footer until interrupted
sky monitor training --mode log     # plain lines
sky log training                    # replay the event log from the start
sky log training -f                 # replay, then follow
sky log training -n 50
sky log export training run.md      # .md or .jsonl
```

The log replays from the beginning, so attaching late loses nothing. A
non-following command stops once the replay goes quiet for `--idle` seconds
(1.0 by default).

## Interactive access

```bash
sky console training                       # login shell on one of the machines
sky console training --node 2
sky console training --command "tail -f /var/log/syslog"
sky repl training                          # Python REPL on the interpreter the node was bootstrapped with
```

The SSH connection belongs to the daemon; the CLI only copies bytes and puts the
local tty in raw mode for the session. The same machine for the whole session.

Jupyter:

```bash
sky notebook install training     # then pick "Skyward (training)" in Jupyter
sky notebook remove training
```

## Scripting it

Every command that prints a table takes `--output json`:

```bash
sky compute list --output json | jq -r '.[] | select(.state=="ready") | .id'
sky offers list --accelerator H100 --output json --limit 0
```

`--output json` means the same thing everywhere: a JSON array of objects keyed
by the table's column names. Table renderings are stringified, so numbers arrive
as strings and an absent value as `"-"` — compare against `"-"`, not `null`.

## Full command reference

`reference/commands.md` has every command with its flags. `sky <command> --help`
is authoritative for the installed build.

## Gotchas

- No configuration file exists. `sky config show` shows what a call *resolved*,
  not what a file said.
- `sky compute create` registers the provider account from *this* process if the
  daemon does not have one, because the daemon never reads the environment. The
  credentials must be exported where `sky` runs.
- A `revision_conflict` on scale or delete is retried automatically (the compute
  is re-read and the write re-sent, five times) — a real conflict means two
  writers, not a stale read.
- `exec` reaches a node whose worker is training: it is a separate SSH channel,
  not a queued task.
- `sky log export` refuses any suffix but `.jsonl` and `.md`.
