# Jupyter notebooks

Skyward can run a Jupyter kernel on an existing compute while Jupyter stays on the local machine. Notebook cells execute remotely; the notebook file, outputs, and local Jupyter extensions remain local.

## Installation

Install the client-side provisioner:

```bash
pip install "skyward[notebook]"
```

The compute image also needs `ipykernel`:

```python
import skyward as sky

with sky.Compute(
    provider=sky.AWS(),
    accelerator=sky.accelerators.A100(),
    image=sky.Image(pip=["ipykernel"]),
    name="research",
    delete_on_exit=False,
) as compute:
    ...
```

The provisioner attaches to the named compute. It does not create or resize one. The compute must be ready before the kernel starts.

## Install a kernel

Use the CLI:

```bash
sky notebook install research
sky notebook install research --url http://127.0.0.1:17590
```

The installed kernel is named `skyward-research` and appears in Jupyter as **Skyward (research)**. The compute argument is a name or id. With no `--url`, the CLI records `SKYWARD_URL` when it is set during installation. If no URL is recorded, the provisioner resolves `SKYWARD_URL` when Jupyter starts and otherwise uses the daemon at `http://127.0.0.1:17590`.

To write the kernelspec into a specific directory instead of the user-level Jupyter location:

```bash
sky notebook install research --directory ./kernels --output json
```

## Remove a kernel

```bash
sky notebook remove research
sky notebook remove research --directory ./kernels
```

`remove` deletes the kernelspec. It does not delete the compute.

## Kernel lifecycle

The provisioner starts a streaming task on the compute. That task launches `ipykernel`, waits for its connection channels, and forwards the five Jupyter channels through the daemon. No local SSH connection is opened.

The current provisioner requires exactly one ready node. Use the distributed function API for multi-node execution.

- Interrupt uses Jupyter's message-based interrupt mode.
- Restart ends the remote kernel stream and starts a new one.
- Shutdown ends the kernel and forwarding, but leaves the compute running.
- Python state in the kernel is lost on restart.

Delete the compute separately when it is no longer needed:

```bash
sky compute delete research
```

## Troubleshooting

If the kernel cannot start, check these conditions:

- `skyward[notebook]` is installed in the local Jupyter environment;
- `ipykernel` is present in the compute image;
- the compute exists and is ready;
- the kernelspec points to the correct daemon with `--url` or `SKYWARD_URL`.

## Next steps

- **[CLI](cli.md)** — managing the compute behind the kernel
- **[Getting started](getting-started.md)** — installation and credential setup
- **[Providers](providers.md)** — choosing what the kernel runs on
