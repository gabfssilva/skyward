# Configuration

Skyward v2 does not read a configuration file. The control plane target is resolved for each client or CLI call.

## Control plane resolution

The resolution order is:

1. an explicit `url` or `--url`;
2. `SKYWARD_URL`;
3. an embedded daemon in the current process.

The embedded daemon stores its SQLite database at `~/.skyward/skyward.sqlite` by default. Pass `database=` to `Compute` or `--database` to the CLI to select another path. A database path is ignored when a remote URL is selected.

```python
import skyward as sky

# Embedded daemon, using ~/.skyward/skyward.sqlite.
with sky.Compute(provider=sky.Container()) as compute:
    result = train(data) >> compute

# Remote daemon.
with sky.Compute(provider=sky.AWS(), url="http://127.0.0.1:7590") as compute:
    result = train(data) >> compute
```

The `Compute` client and the CLI use the same resolution rules. Inspect the resolved values with:

```bash
sky config path
sky config show
sky config validate
```

## Provider accounts

Provider factories resolve credentials in the client process. A provider descriptor contains the provider kind, its non-secret configuration, credentials, and an optional account name. When a compute starts, the descriptor registers the named account with the daemon if it does not exist. Credentials are not returned by provider read operations.

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
