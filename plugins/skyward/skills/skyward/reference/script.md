# Files that declare their compute: `sky run`

`sky run FILE [ARGS...]` runs a local Python file on the compute the file itself declares. It creates that compute or attaches to it, queues the run on it and returns at once; the daemon holds the compute for the run and, by default, deletes it once the run ends. `-f`/`--foreground` stays for the run instead (below). A file declares its compute in one of two ways, each with a reference of its own:

| | PEP 723 header (`reference/script-header.md`) | `sky.app` (`reference/script-app.md`) |
|---|---|---|
| the compute is declared | in a `[tool.skyward]` table of the `# /// script` block | in Python: `sky.app(...)`, with the arguments `sky.Compute` takes |
| what runs on the node | the whole file, as `__main__`, `ARGS` as `sys.argv[1:]` | one decorated function, `ARGS` parsed against its signature |
| what comes back (`--foreground`, `sky compute result`) | stdout and the exit status | stdout, the exit status, and the return value as JSON |
| the image | `requires-python`, `dependencies`, `[tool.skyward.image]` | `image=sky.Image(...)` |
| the provider | a kind, with that kind's default settings | a provider struct and its settings: `sky.RunPod(cloud_type="community")` |
| on this machine | nothing of the file runs | the file is imported to find its functions, so its top-level imports must be installed here |

A file with a `[tool.skyward]` table is a header script, whatever else it holds.

The header fits a file that is already a script: it reads `sys.argv`, prints, exits. `sky.app` fits typed entry points, a result that comes back, several commands in one file, or provider settings a header cannot express; a decorated function is still a plain function when called from Python, so the file keeps working locally.

## On the node

The file is sent as text and executed inside a worker on the node, as a task, the same way a `@sky.function` would be: the image, the plugins and the runtime API are around it. The code that runs there uses the node-side API (`sky.instance_info()`, `sky.shard`, `sky.is_head()`, `sky.dict`/`sky.barrier`/..., output policy) and never opens a `sky.Compute`. The docs site's `reference/runtime/` and `distributed-collections/` pages cover those (the site map is `reference/sdk.md`).

- The file's text travels, and with it what its image includes (below). Any other local module it imports, or a local path it opens, does not exist on the node. Data goes through `sky compute upload` (on a compute kept with `delete_on_exit` false), object storage, or a download inside the code.
- The exit status: `sys.exit(n)` is `n`; finishing, or `sys.exit()`, is `0`; `sys.exit("message")` prints the message to stderr and is `1`; an uncaught exception prints its traceback and is `1`. `sky run --foreground` and `sky compute result` exit with the worst node's status.

## Local code: the image's `includes`

A local package the file imports, one not published anywhere, goes in the image's `includes`: `includes = [...]` under `[tool.skyward.image]` in a header, `image=sky.Image(includes=[...])` in a `sky.app`. `excludes` sits beside it.

```python
gpu = sky.app(provider=sky.AWS(), image=sky.Image(pip=["numpy"], includes=["src/classy_enc"]))


@gpu
def train(epochs: int = 10) -> None:
    from classy_enc.train import main

    main(epochs)
```

- A relative path counts from the file's directory, not from where `sky run` was typed; an absolute one is taken as it is.
- Each path lands under its own name, in a directory that is first on the node's `sys.path` while the file runs: `src/classy_enc` is `import classy_enc`, `helpers.py` is `import helpers`. Being first, it wins over an installed package of the same name.
- A directory is walked. `__pycache__`, `*.pyc`, `.git`, `.venv`, `node_modules`, `*.egg-info`, and whatever `excludes` names (glob patterns, matched against each component of a path) are left out.
- It is packed on this machine and sent with **every run**, not installed when a machine is set up, whether or not the image is `mutable`. A run that attaches to a compute kept up runs the code as it is now. The machines are built without it, so `sky compute run`, `exec` and `ssh` on that compute do not see it.
- It is on the path of the process running the file, and of what that process forks; a new interpreter the code starts itself (`subprocess.run([sys.executable, ...])`) does not have it.
- A path that is not there, or two paths that would land under one name (`a/utils` and `b/utils`), are refused before anything is bought.

## Validation happens before a machine is bought

Everything the file declares is checked before the daemon is contacted: the header's fields, plugins and bounds, or `sky.app`'s command line against the function's signature. A mistake is a one-line refusal and a non-zero exit, with nothing billed. A valid file with no daemon stops with `no daemon at ... — run: sky server start`: `sky run` does not start one.

For a first run without a bill, use the `container` provider: the nodes are local Docker containers. Switching the provider afterwards is a different compute (below), so nothing carries over from the smoke test.

## The compute it runs on

The compute is named `<file stem>-<8 hex digits>`, the digits a digest of what would take other machines: the provider, the accelerator, `cpus`, `memory_gb`, `region`, `allocation`, the image and the plugins. The image's `includes` and `excludes` count as the paths written, not as what the files hold: adding or renaming an include names another compute, editing an included file does not. An image declared `mutable` (`mutable = true` under `[tool.skyward.image]`, `image=sky.Image(mutable=True)` in a `sky.app`) is left out of the digest: its packages can change without naming another compute, and the next run against the compute that is up patches `pip` and `pip_indexes` on it before submitting the work. Includes still travel with each run and are not part of the patch. `nodes`, `delete_on_exit` and a `sky.app`'s `options` are not in it, and neither are credentials, so a rotated key names nothing new. The two forms count the provider and the accelerator differently:

- **header:** the provider by its kind, the accelerator as written. `"RTX_3090"` and `"rtx-3090"` are two computes.
- **`sky.app`:** the provider by its kind plus whichever settings differ from that kind's defaults, so a default changed by a release renames nothing. The accelerator counts by what it resolves to: `"A100"`, `"a100"` and `sky.accelerators.A100()` are one compute, `sky.accelerators.A100(count=2)` another.

| on `sky run` | happens |
|---|---|
| no compute by that name, or it is `deleted` | a new one is created, and the run's task waits in its queue for the machines |
| one is up | the run attaches to it; nothing is bootstrapped again |
| one is up and `nodes` changed | it is resized first, then the work is submitted |
| one is up, its image is `mutable`, and `pip` or `pip_indexes` changed | the image is patched, and the nodes refresh before they take the work |
| one is held by another run: its task has not settled, or a `--foreground` run is on it | refused, `lease_held`, with nothing queued |
| one is up, `nodes` changed, and it runs a collective plugin (`torch`, `jax`, `accelerate`) | refused, `compute_not_resizable`: delete it and run again |
| one is `deleting` | refused: `run again once it is gone` |

- **Changing anything in the digest names another compute.** A previous one kept with `delete_on_exit` false stays up and billing under its old name. `sky compute list` shows it; `sky compute delete <name>` ends it.
- With `delete_on_exit` true (the default) the compute is deleted when the run ends, so every run pays the provisioning and bootstrap again. For iterating, set it to false: the compute stays after the run, and the next run attaches instead of provisioning.
- The compute is an ordinary compute: `sky compute view <name>`, `sky log <name>`, `sky compute exec/ssh <name>` all reach it (`reference/cli.md`, `reference/nodes.md`).

## Provider account

The run registers the provider account under its name (the kind, unless the struct sets `name`) from **this process's environment**: the credential variables in `reference/cli.md`. `sky.Compute(provider=...)` does the same. The settings registered are the declared ones: a header's kind brings that kind's defaults, a `sky.app` brings its struct's fields. An account of that name that `sky providers set KIND --config ...` gave other settings is overwritten with them.

## Returning at once, or `--foreground`

By default `sky run` waits on nothing, neither the machines nor the run. It asks for the compute, queues the run's task on it, and holds the compute's lease for that task, so no local process has to stay alive: the daemon keeps the compute while the task is queued or running and, once it settles, deletes the compute (cause `finished`) unless `delete_on_exit` is false. Then it prints where the run is, and exits 0:

```
compute  train-1a2b3c4d (requested)
task     tsk_9f8e7d6c5b4a
follow   sky monitor train-1a2b3c4d
result   sky compute result tsk_9f8e7d6c5b4a
after    deleted once the task settles
```

- `sky monitor <name>` follows the run from then on, a line per event on stdout — a machine changing state, a line the code printed, the task's attempt ending (`tsk_… │ succeeded`) — and ends once the compute is deleted. An assistant runs it as a background monitor (Claude Code's Monitor tool), so each line reaches it as a notification; `sky log <name>` replays what came before, and `sky compute tasks <name>` shows the task's state.
- `sky compute result <task>` waits for the task to settle, prints a `sky.app` function's return value as JSON (a list by rank under `--node all`), and exits with the worst node's status. A task that failed or was cancelled is refused with its reason.
- What goes wrong while the machines are bought or bootstrapped is in `sky log` and `sky compute get`, not in this command's exit status.
- `--node all` queued before any machine is ready is for every rank the compute opens with, each waiting for its machine.

`-f`/`--foreground` stays for the run: it draws the compute while it comes up, prints what the code prints and what a function returned, and exits with the worst node's status. Interrupted, or once the run ends, it deletes the compute unless `delete_on_exit` is false.

## Output under `--foreground`

- **stdout** carries the lines the code printed, each after the rank of the node that printed it (`0 │ rank 0 of 2 ...`); the lines of a broadcast arrive interleaved. Under `sky.app`, the return value follows as the last line.
- **stderr** carries the compute's events (machines requested, bootstrapping, ready, ...), rendered the way the console renders them, plus the few things `sky run` itself says.

So `sky run -f train.py > out.txt` keeps what the code wrote and leaves the progress on the terminal.

## `--node`

| `--node` | runs on |
|---|---|
| `any` (default) | one node with a slot free |
| `all` | every node, once each |
| a rank (`0`, `1`, ...) | that node |

A multi-node job (code that calls `sky.shard`, forms a process group, or waits on a `sky.barrier`) needs `--node all`: with `any` only one rank runs it. `--node`, `--url` and `-f`/`--foreground` belong to `sky run` wherever they are on the line; everything else goes to the file, and after `--` everything does: `sky run train.py -- -f config.yaml`.

## `sky run`, `sky compute run` or the SDK

| want | use |
|---|---|
| the file declares its machines, one command does everything | `sky run FILE` |
| run a file on a compute that already exists (made by `sky compute create`, the SDK, or an earlier `sky run`) | `sky compute run REF FILE`, which runs the whole file and reads no declaration (`reference/nodes.md`) |
| several functions in one program, results back as Python values, `map`, futures, streaming, fallback across providers | the SDK (`reference/sdk.md`) |
