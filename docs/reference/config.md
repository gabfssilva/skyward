# Configuration

Skyward does not read a configuration file. The control plane target is resolved for each client or CLI call.

## Control plane resolution

The resolution order is:

1. an explicit `url` or `--url`;
2. `SKYWARD_URL`;
3. the daemon at `http://127.0.0.1:17590`.

`Compute` starts a daemon at that address when none answers, printing `no server is running, starting it now`. The daemon detaches, so it outlives the process that started it and goes on reconciling the machines it bought. Stop it with `sky server stop`. The CLI never starts one: it reports that nothing answers.

A daemon is refused when it runs a different version of Skyward than the client, because the same routes carry other wire types. Stop it and let the client start one, or point the client at a daemon on its version.

Passing `database=` to `Compute` runs the control plane inside the current process over that file instead of reaching a daemon. `sky server start --database` gives a daemon its own. Either way the default is `~/.skyward/skyward.sqlite`, and a database is ignored when a URL is given.

```python
import skyward as sky

# The daemon at 127.0.0.1:17590, started here if nobody has.
with sky.Compute(provider=sky.Container()) as compute:
    result = train(data) >> compute

# The control plane in this process, over a database of its own.
with sky.Compute(provider=sky.Container(), database="/tmp/experiment.sqlite") as compute:
    result = train(data) >> compute

# Remote daemon.
with sky.Compute(provider=sky.AWS(), url="http://127.0.0.1:17590") as compute:
    result = train(data) >> compute
```

The `Compute` client and the CLI use the same resolution rules. Inspect the resolved values with:

```bash
sky config path
sky config show
sky config validate
```

## Provider accounts

Provider accounts resolve credentials in the client process. A provider descriptor contains the provider kind, its non-secret configuration, credentials, and an optional account name. When a compute starts, the descriptor registers the named account with the daemon if it does not exist. Credentials are not returned by provider read operations.

```python
provider = sky.AWS(name="production", region="us-east-1")

with sky.Compute(provider=provider, accelerator="A100") as compute:
    train(data) >> compute
```

Use a distinct `name` for multiple accounts of the same provider kind. Provider configuration belongs to the account, not to a global singleton.

## Compute configuration

The public configuration objects are composed at the `Compute` boundary:

- `Spec` describes one provider and hardware alternative;
- `Options` controls timeouts, retries, health checks, and autoscaling;
- `Executor` controls task execution on each node;
- `Image`, `Volume`, and `Port` describe the node environment and local tunnels.

See [Compute and task dispatch](pool.md) for the full Python surface and [Cloud providers](../providers.md) for account credentials and offer caching.

::: skyward.Provider

::: skyward.Image
