# Using the Skyward CLI

`sky` is a thin client over the Skyward daemon's HTTP API. It decides nothing and stores nothing: every command turns a few flags into one or two HTTP calls and prints what came back. Everything that matters — computes, nodes, offers, provider accounts, event logs — lives in the daemon.

This file covers the daemon, provider accounts, offers and the lifecycle of a compute. Work on the nodes of a compute that is up (`exec`, `compute run`, files, `ssh`, `repl`, notebook) is `reference/nodes.md`; `sky run` is `reference/script.md`.

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
sky server restart        # stop it, then start one in its place
sky server interface set tailscale0   # also listen there, from the next start; loopback stays
```

A command that reaches nothing says so and stops:

```
$ sky compute list
no daemon at http://127.0.0.1:17590 — run: sky server start
```

The SDK is the other thing that starts one. A `sky.Compute` that names no daemon looks at the same address, starts a daemon there when nothing answers, and leaves it running — so a daemon, and computes in it, can be there without anyone having run `sky server start`. It is the same daemon and the same pidfile: `start` then says `already running (pid N)`, and `stop` ends it whichever of the two began it. An SDK client refusing to talk to it (`the daemon at ... runs skyward X, this process runs skyward Y`) is a daemon left over from another version — `sky server stop`, and the next run starts one of its own.

`sky server start --foreground` stays attached, for a dev loop. `--database` belongs to `sky server start` alone — no other command takes one, because no other command owns state.

Daemon logs go to `~/.skyward/server.log`. `sky server start --database PATH` points it at a different SQLite file (it passes `SKYWARD_DATABASE` to the process). `sky server stop` signals the recorded pid — there is no shutdown endpoint, deliberately.

## Provider accounts

A provider is a *registered account*, not a kind. The daemon holds it and is the only thing that ever uses it.

```bash
sky providers list --kinds        # what can be registered, and which credentials each needs
sky providers set runpod          # register the account
sky providers set runpod --config cloud_type=community --config region=EU
sky providers set aws --config region=eu-west-1 --name aws-eu
sky providers list                # what is registered
sky providers check               # did the daemon's last use of each account work?
```

**Credentials are never passed on the command line and cannot be.** `sky providers set` reads them from the environment the same way the SDK does, so a key never lands in shell history. Export them first:

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

`--config key=value` takes the account's own fields and is validated against the account struct before anything is sent, so a misspelled setting is refused here rather than at provisioning time. `--name` registers under a name other than the kind; a compute created without a name looks for the kind.

Use `container` for a local smoke test: it runs nodes as Docker containers, no cloud and no bill.

## Offers and prices

One GET against the daemon's catalog. The daemon owns the per-provider TTL and the refresh a stale provider triggers.

```bash
sky offers list --accelerator H100 --min-count 8 --limit 10
sky offers list --provider runpod --max-price 2.5 --limit 0     # 0 prints everything
sky offers summary --accelerator A100                           # cheapest/average/dearest per provider
sky offers fetch                                                # force a refetch, report counts
```

An empty listing on a daemon with no registered accounts is not an answer about hardware — the CLI says so on stderr. Register an account first.

## Computes

```bash
sky compute create --provider runpod --accelerator A100 --nodes 4 --name training
sky compute list
sky compute get training
sky compute view training          # the compute plus the machines it stands on
sky compute scale training --nodes 8
sky compute scale training --nodes 2:8      # elastic range, MIN:MAX
sky compute update training --pip six       # mutable image only: pip, pip-index, include, exclude
sky compute delete training
```

`create` also takes the image the nodes build, with the meaning `sky.Image` gives the same fields: `--base`, `--python`, `--pip`, `--apt`, `--pip-index`, `--env KEY=VALUE` (each list flag repeats), and `--plugin NAME` or `--plugin NAME:key=value,...` (`--plugin torch:backend=gloo`), checked against the plugin's own fields before anything is sent. `sky compute view` shows the image a compute asked for.

```bash
sky compute create --provider runpod --accelerator H100 --name ft \
  --python 3.12 --pip torch --pip transformers --apt build-essential --plugin torch
```

`sky new` is an alias for `sky compute create`. `sky status [ref]`, `sky sessions` and `sky stop <ref>` are the same commands under the names you reach for when the question is "what is running".

Three things to expect:

- **create returns immediately.** It posts intent; the machines arrive afterwards. Watch with `sky monitor <ref>` or `sky log <ref> -f`.
- **delete is accepted, not done.** What comes back is still `deleting`, and stays that way until the provider confirms the machines are gone.
- **scale returns a new `generation`**, not a finished resize. What is up is kept; the difference is bought or drained by reconciliation.
- **update changes the mutable fields of a running compute's image** (`--pip`, `--pip-index`, `--include`, `--exclude`) and is refused with `image_fixed` when the image was not created `mutable`. The flags give the new lists, and the nodes refresh afterwards.

`--ttl SECONDS` on create is the dead-man switch the providers that support one arm on each machine: with nobody connected for that long, the machine removes itself rather than billing for a daemon that is never coming back. `--ttl 0` never does.

### When `provisioning` does not move

A provider that has no stock refuses the launch and the reconciler simply tries again, so a compute can sit in `provisioning` for many minutes with nothing wrong. **That refusal is not in the compute's event log.** `sky log` carries `node.requested`, `node.connecting`, `node.bootstrapping`, `node.phase`, `node.console`, `node.ready` and `compute.cost` — the provider exception only reaches the daemon's own log:

```bash
tail -f ~/.skyward/server.log          # where a failed launch actually says why
sky log <id> -f                        # where the node's own progress is
```

A stock-out reads like `no market could place a runpod machine` wrapping the provider's message. It is worth waiting through — retries do land.

## Watching a compute

```bash
sky monitor training                # live Rich footer until interrupted
sky monitor training --mode log     # plain lines
sky log training                    # replay the event log from the start
sky log training -f                 # replay, then follow
sky log training -n 50
sky log export training run.md      # .md or .jsonl
```

The log replays from the beginning, so attaching late loses nothing. A non-following command stops once the replay goes quiet for `--idle` seconds (1.0 by default).

`sky app` watches every live compute on one screen, until `q`.

## JSON output

Every command that prints a table takes `--output json`:

```bash
sky compute list --output json | jq -r '.[] | select(.state=="ready") | .id'
sky offers list --accelerator H100 --output json --limit 0
```

`--output json` means the same thing everywhere: a JSON array of objects keyed by the table's column names. Table renderings are stringified, so numbers arrive as strings and an absent value as `"-"` — compare against `"-"`, not `null`.

## Full command reference

`reference/commands.md` has every command with its flags. `sky <command> --help` is authoritative for the installed build.

## Gotchas

- No configuration file exists. `sky config show` shows what a call *resolved*, not what a file said.
- `sky compute create` registers the provider account from *this* process if the daemon does not have one, because the daemon never reads the environment. The credentials must be exported where `sky` runs.
- A `revision_conflict` on scale or delete is retried automatically (the compute is re-read and the write re-sent, five times) — a real conflict means two writers, not a stale read.
- `sky log export` refuses any suffix but `.jsonl` and `.md`.
