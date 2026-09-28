# `sky.app` declares the compute

`sky.app(...)` takes what `sky.Compute` is told about the machines, as a value and without the pool. The functions it decorates are the file's commands: `sky run FILE [COMMAND] [ARGS...]` parses `ARGS` against the chosen function's signature, runs the function on the compute, and prints what it returned as JSON. What every `sky run` shares (the node, the compute's name, the account, output, `--node`) is `reference/script.md`.

```python
from pathlib import Path

import skyward as sky

gpu = sky.app(
    provider=sky.RunPod(cloud_type="community"),
    accelerator=sky.accelerators.A100(),
    nodes=2,
    image=sky.Image(pip=["numpy"]),
    delete_on_exit=False,
)


@gpu
def train(data: Path, epochs: int = 10, lr: float = 1e-3) -> dict[str, float]:
    """Train on DATA for EPOCHS."""
    import numpy as np

    return {"epochs": epochs, "loss": float(np.exp(-epochs * lr))}


@gpu
def evaluate_model(checkpoint: str) -> float:
    return 0.5


if __name__ == "__main__":
    print(train(Path("local.csv"), epochs=1))
```

```bash
sky run train.py --help                                # the file's commands, with their docstrings
sky run train.py train --help                          # one command's parameters
sky run train.py train data.csv --epochs 5 --node all  # prints [{"epochs":5,...},{"epochs":5,...}], one per rank
sky run train.py evaluate-model ckpt.pt                # prints 0.5
```

## What `sky.app` takes

The arguments `sky.Compute` takes for the machines, with the same meaning: `provider` (required, a provider struct: `sky.AWS(...)`, `sky.RunPod(...)`, ...), `accelerator` (a string or `sky.accelerators.*`), `cpus`, `memory_gb`, `region`, `nodes` (`4`, `(2, 8)`, `sky.Nodes(...)`), `allocation`, `image` (`sky.Image(...)`: `python`, `pip`, `apt`, `env`, `pip_indexes`, `base`, `skyward`), `plugins` (`sky.plugins.Torch()`, ...), `delete_on_exit`. Nothing else: no `ttl`, `executor`, `options`, `ports` or `name` (the compute's name is derived, see `reference/script.md`). The SDK reference (`reference/sdk.md`) covers each of them.

It can decorate directly (`@sky.app(provider=...)`) or be bound once and reused (`gpu = sky.app(...)`, then `@gpu`). Functions under one `sky.app` share one compute, and so do functions whose `sky.app`s declare the same machines (`nodes` and `delete_on_exit` aside); one that declares other machines runs on another compute.

A PEP 723 block without a `[tool.skyward]` table is not read: its `dependencies` do not reach the image. Packages go in `image=sky.Image(pip=[...])`.

## Commands

- The file's commands are the decorated functions it **defines**. One it imports belongs to the file that defined it and is not a command here.
- A command is named after its function, `_` spelled `-`: `evaluate_model` is `evaluate-model`.
- With one command, naming it is optional: `sky run train.py data.csv`. With more, the name comes first, and a missing or unknown one is refused with the list of commands.
- The parameters are parsed the way a command line parses a signature: positional or `--name value`, a list as repeated values, a `bool` as a `--flag`, a dict as `--name.key value`, each converted to its annotation. A value that does not convert is refused before anything is bought. The docstring's first line is the command's help.
- A parameter named `node` or `url` is refused: `--node` and `--url` belong to `sky run`, wherever they are on the line.
- The arguments travel to the node as msgpack and are converted back against the same annotations there: strings, numbers, booleans, lists, dicts and `Path` arrive as themselves. An argument msgpack cannot carry is refused: `an argument cannot travel to the node`.

## What comes back

- The return value is printed on stdout as JSON once the function returns: the one value, or under `--node all` a JSON list in rank order. A value JSON has no form for is written as its `str`.
- Nothing is printed when the function returns `None`, nor when a node did not return: its traceback is already on the terminal, and the exit status says so.
- Returning is status `0`; `sys.exit` and exceptions end the call with the status `reference/script.md` gives them.
- Lines the function printed come first, each after its rank; the JSON is the last line of stdout.

## Where the file runs

The file runs twice, and neither run is as `__main__`:

1. **Here**, when `sky run` imports it to find the commands and parse the command line. Its top level executes on this machine, so every top-level import must be installed where `sky run` runs; a missing one fails with `ModuleNotFoundError` before a machine is bought. An import only the node has (`torch` on a laptop, say) goes inside the function.
2. **On the node**, as a module named after the file (`train`), before the function is called with the arguments.

So an `if __name__ == "__main__":` block runs in neither, and the file stays usable by hand: called from Python, a decorated function is only the function, and runs where it is called (`python train.py` above calls `train` on this machine).

On the node skyward is installed without its client. The file's top level may use `sky.app`, provider structs, `sky.accelerators`, `sky.Image`, `sky.plugins` and `@sky.function`; touching `sky.Compute` there raises `ImportError`.
