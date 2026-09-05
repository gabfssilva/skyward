# Salad Cloud

`sky.Salad` uses Salad Container Engine. The adapter creates one container group per node, each with a single replica.

The adapter needs `skyward[salad]`, which brings salad-cloud-sdk and websockets. Without it the daemon does not register the kind.

```python
import skyward as sky

provider = sky.Salad(
    api_key="...",
    organization="my-org",
    project="training",
    priority="low",
)
```

The API key can come from `SALAD_API_KEY`. `organization` and `project` can come from `SALAD_ORGANIZATION` and `SALAD_PROJECT`.

| Parameter | Default | Meaning |
|---|---|---|
| `api_key` | `SALAD_API_KEY` | Account credential. |
| `organization` | `SALAD_ORGANIZATION` | Salad organization name. |
| `project` | `SALAD_PROJECT` | Salad project name, lowercased. |
| `priority` | `"low"` | `high`, `medium`, `low` or `batch`. Selects the price and how readily the workload is preempted. |
| `country_codes` | every country | Restrict placement to these ISO country codes. |
| `image` | a CUDA runtime image | Base image, overridden by `Image(base=...)`. An image that already carries sshd, curl and websocat is used as is; otherwise it must be Debian or Ubuntu based, because the container command installs them with `apt-get`. |
| `cpus` | `4` | vCPUs per node. |
| `memory_gb` | `16` | RAM per node, in whole GiB up to 60. Salad quotes no size with a GPU class, so this is what the offer advertises and what the container is created with. |
| `storage_gb` | `50` | Container storage. Salad's floor is 1 GiB. |
| `request_timeout` | `30` | Seconds for one Salad API call. |

Salad prices GPU classes by container priority. Skyward exposes the selected priority as an on-demand offer; it is billed per second. `spot` allocation is not available.

## How a node is reached

Salad gives a container one way in: the Container Gateway, an HTTP reverse proxy in front of a single port. There is no inbound TCP, and the SSH relay shown in the Salad portal is a preview feature that runs outside the container — on a node whose relay fails it closes the connection before the SSH banner, and only reallocating the instance clears it.

The adapter does not use that relay. Each container runs sshd behind a WebSocket-to-TCP bridge on the gateway port, and the daemon runs one loopback listener per node whose connections become WebSockets to that node's gateway domain. Everything above the adapter dials an ordinary host and port.

Two consequences follow from the gateway:

- Its domain name addresses a container group and load balances across that group's replicas, so a node is its own group of one. That is also what makes a single node individually terminable.
- It cannot carry the `Salad-Api-Key` header on a WebSocket upgrade, so the gateway is opened unauthenticated. A node is protected by its unguessable domain name and by sshd accepting Skyward's ephemeral key and nothing else.

Nodes still have no way to reach each other, so cluster formation is rejected: a Salad compute is a fleet of independent nodes. Leave `options.cluster` unset or `False`.
