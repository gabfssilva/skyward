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

The API key can come from `SALAD_API_KEY`. `organization` and `project` can come from `SALAD_ORGANIZATION` and `SALAD_PROJECT`; the project name is lowercased.

`priority` (`high`, `medium`, `low` or `batch`) selects the price and how readily the workload is preempted. `spot` allocation is not available; every offer is on-demand, billed per second.

The base image is a CUDA runtime image unless `image` or `Image(base=...)` names another. An image that already carries sshd, curl and websocat is used as is; otherwise it must be Debian or Ubuntu based, because the container command installs them with `apt-get`. `storage_gb` is the container's storage, and Salad's floor is 1 GiB.

`min_download_mbps` and `min_upload_mbps` hold a node to a download and an upload speed. Salad has no filter for either — a node is somebody's home computer, on whatever connection their ISP sold them, and a home connection usually uploads at a fraction of what it downloads, so each direction has its own floor. When one is set, the container measures that direction against Cloudflare's speed test before sshd comes up, three attempts twenty seconds apart, download first, and a node short of it on every attempt asks Salad to reallocate the replica. An attempt is eight transfers at once, of at most fifteen seconds each, and the speed is what they moved together over the time they took: one TCP connection carries a fraction of a link, so one transfer would measure the connection rather than the node. What the check prints is the node's progress while it boots, and the daemon's log keeps it. The reallocation moves it to another machine and keeps the rejected one out of this account's pool for 48 hours, which is the cost of the knob: set the speed the workload needs, not the one it would like. A speed test that does not answer is not held against the node. Unset, that direction is not measured.

::: skyward.Salad

## What the account is offered

Salad sells nodes with a GPU and nodes without, and prices only the first. The adapter offers both.

A **GPU class** carries its own price, one per container priority, and Salad bills no vCPUs or RAM beside it — the quoted price is the whole node, at any size. A class is therefore one offer, published at the largest group the adapter sells (16 vCPUs, 32 GiB) so that a compute asking for any size up to it is sold the class.

A **CPU-only node** is the same container group without a GPU class, billed per vCPU-hour and per GB-hour. Salad sizes a container group from the request that creates it and quotes no CPU catalog over its API, so every size between 1 and 16 vCPUs and between 1 and 32 GiB is offered — 512 of them, priced `cpus × vcpu_price + memory_gb × memory_price`. Nothing is rounded to a list of sizes somebody chose: asking for 5 vCPUs and 7 GiB buys 5 vCPUs and 7 GiB. The default rates are Salad's published ones as of September 2026; **check them against your account**, since nothing in the API quotes them back.

`priority` selects a GPU class's price. Salad's published CPU-only rates do not vary by priority, and its pricing page states CPU-only groups run at the lowest tier; whether an account is ever billed a different CPU rate per priority is not something the API answers, so `vcpu_price` and `memory_price` are single values.

The container is created at the compute's `cpus` and `memory_gb`. What the compute leaves out comes from the offer that was bought: on the CPU-only shelf the size the market picked, which is the cheapest that fits and for a compute that names neither is 1 vCPU and 1 GiB; on a GPU class the largest size, which costs nothing more.

A compute that names no accelerator is sold a CPU node, even where a GPU class would be cheaper — and on Salad one often is, since a class bundles its vCPUs and RAM. The saving is not the whole price: a GPU class places only on machines carrying that GPU, a smaller pool than a CPU node runs on. Name `accelerator` to buy one deliberately.

## How a node is reached

Salad gives a container one way in: the Container Gateway, an HTTP reverse proxy in front of a single port. There is no inbound TCP, and the SSH relay shown in the Salad portal is a preview feature that runs outside the container — on a node whose relay fails it closes the connection before the SSH banner, and only reallocating the instance clears it.

The adapter does not use that relay. Each container runs sshd behind a WebSocket-to-TCP bridge on the gateway port, and the daemon runs one loopback listener per node whose connections become WebSockets to that node's gateway domain. Everything above the adapter dials an ordinary host and port.

Two consequences follow from the gateway:

- Its domain name addresses a container group and load balances across that group's replicas, so a node is its own group of one. That is also what makes a single node individually terminable.
- It cannot carry the `Salad-Api-Key` header on a WebSocket upgrade, so the gateway is opened unauthenticated. A node is protected by its unguessable domain name and by sshd accepting Skyward's ephemeral key and nothing else.

Nodes still have no way to reach each other, so cluster formation is rejected: a Salad compute is a fleet of independent nodes. Leave `options.cluster` unset or `False`.
