---
name: skyward
description: Skyward, the GPU compute orchestrator. Covers Python against the SDK (@sky.function, sky.Compute, the >> @ & > operators, accelerators, providers, plugins, distributed training and collections, volumes, notebook kernel); the `sky` CLI (daemon, provider accounts, offers and GPU prices, creating/scaling/deleting computes, exec, files, ssh and repl on nodes, event logs); and files that declare their own compute, in a PEP 723 header ([tool.skyward]) or with the sky.app decorator, run with `sky run`. Use it whenever the user mentions skyward, sky.function, sky.Compute, sky.app, sky.shard, sky run, sky compute, sky offers, sky providers, sky server or sky log, or wants to provision GPUs, compare GPU prices, spin up nodes or run a script or training job on remote machines, even without naming Skyward.
---

# Skyward

This file is an index. Read the one reference that matches the task, and open another only when the task crosses into it. Paths are relative to this skill's directory.

| the task | read | why |
|---|---|---|
| Python that opens a `sky.Compute` and dispatches `@sky.function`s: operators, accelerators, providers, plugins, distributed training, collections, volumes, notebook | `reference/sdk.md` (a map of the docs site) | All of Skyward: many functions in one program, results back as Python values, `map`, broadcast, futures, streaming, fallback across providers, elastic sizes, volumes, ports. The choice when the orchestration is part of the program. |
| a file that declares its compute and runs with `sky run`: a PEP 723 header (`# /// script`, `[tool.skyward]`) or functions decorated with `sky.app` | `reference/script.md`, then `reference/script-header.md` or `reference/script-app.md` | The shortest path from a file to a GPU: one command, the machines declared next to the code. A header runs the whole file and gives back stdout and an exit status; `sky.app` makes functions into typed commands whose return value comes back as JSON. Either way one compute per run, and no `ttl`, executor or ports; local packages go in the image's `includes`, sent with every run. |
| the daemon, provider accounts and credentials, offers and prices, creating/scaling/deleting a compute, watching it, `--output json` | `reference/cli.md` | Operating by hand, without code: register accounts, compare prices, keep a compute up and watch it. |
| work on the nodes of a compute that is up: `exec`, `compute run`, files, `ssh`, `repl`, Jupyter, what is on the node | `reference/nodes.md` | Reaching machines that already exist, to debug a node, move files or run a file on a compute made elsewhere. What is changed here is gone with the machine; what a node needs belongs in its image. |
| one command's exact flags, columns or defaults | `reference/commands.md` | Lookup: every command and flag in one place. |

Don't write the API or a command line from memory; it moves. When a reference and the installed build disagree, the build wins: `sky version`, `sky <command> --help`, `python -c "import skyward as sky; help(sky.Compute)"`.
