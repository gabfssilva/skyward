# Choosing the best provider: an opinionated guide

This is not a feature comparison or a benchmark page. This is what I've learned spending real money on GPU cloud over the past months — which providers I reach for and why. Your mileage will vary depending on workload, region, and budget, but these are the patterns that have held up for me.

If you're looking for setup instructions and parameter tables, see the [Providers](providers.md) reference.

## My daily drivers

I use two providers for almost everything: **RunPod** for experimentation and **AWS** for production workloads.

**RunPod** is where I prototype. The pricing is competitive, the catalog is wide enough that I can test across different GPU tiers without committing to anything, and the setup is trivial — one API key. When I need a quick A100 or a cheap RTX 3090 to validate an idea, RunPod is the first thing I reach for. It's also what I recommend to anyone getting started with GPU cloud — no IAM policies, no VPC configuration. Just an API key and you're running.

**AWS** is where I run anything that needs to be reliable. Provisioning is the fastest of any provider I've used, and it's not close. AWS uses EC2 Fleet to launch instances in batch — one API call provisions your entire cluster. Most other providers provision instances one at a time through individual API calls, which is visibly slower. More importantly, AWS instances rarely have the infrastructure issues that plague smaller providers: flaky networking, port problems, stale drivers. When I'm running a multi-hour training job and can't afford to babysit it, I use AWS.

The split isn't just about reliability vs cost. It's also about hardware:

- **RunPod** for RTX 3090, RTX 4090, RTX 5090, and A100s at competitive prices
- **AWS** for T4, T4g (ARM), L4, L40S with excellent spot pricing

## Provider by provider

### RunPod

My go-to for experimentation. The Community Cloud tier gives access to consumer GPUs at prices that are hard to beat elsewhere. RTX 3090s in particular are extremely cost-effective — they have 24 GB of VRAM, solid FP32 performance, and Community Cloud pricing makes them one of the cheapest ways to run real GPU workloads.

RTX 4090s are the next step up when you need more compute. The RTX 5090 is useful for pipelines that need a bit more VRAM than a 4090 provides, without jumping to datacenter hardware pricing.

One thing I've learned: **use spot instances with a bid multiplier between 1.2x and 1.4x**. The default bid often gets you preempted quickly. A 20-40% premium over the minimum bid price dramatically reduces preemption frequency, and you're still paying far less than on-demand.

```python
sky.Compute(
    provider=sky.RunPod(bid_multiplier=1.3),
    accelerator=sky.accelerators.RTX_3090(),
    allocation="spot",
)
```

One caveat for multi-node training: RunPod's individual pods don't share a private network. Each pod gets its own IP, but pods can't reach each other directly. Skyward handles this transparently for task dispatch (everything goes over SSH), but if your workload needs inter-node communication (NCCL for distributed training), you'll need RunPod's global networking — which only works with NVIDIA GPUs on on-demand instances. Otherwise, run in standalone mode with `options=sky.Options(cluster=False)`.

### AWS

The most stable provider, period. I've had fewer infrastructure issues with AWS than with any other provider. Networking works, ports are reachable, drivers are current. When something goes wrong with a cloud workload, it's almost never the AWS side.

The T4g (Arm-based T4) is really cool for smaller models and very light training. You get a significant boost from a regular CPU, but still spend around $0.14/hr spot for a 4-vCPU instance. Spot instances on AWS are also remarkably stable — they rarely get reclaimed compared to other providers. The regular T4, L4, and L40S are also solid choices with good spot pricing.

AWS provisioning speed is in a league of its own. EC2 Fleet provisions your entire cluster in a single API call — all instances launch in parallel, in the same availability zone, with networking already configured through VPC. Other providers make one API call per instance and rely on external networking layers. This is why AWS goes from zero to SSH-ready faster than anything else.

The other advantage for multi-node work: AWS instances share a VPC by default. No overlay networks, no special configuration — nodes can reach each other directly. This makes distributed training (NCCL, Gloo) more stable compared to providers that need overlay networking or SSH tunneling for inter-node communication.

The downside is the setup overhead. IAM policies, VPCs, security groups — there's more configuration to get right compared to providers that just need an API key. But once it's set up, it stays out of your way.

### Vast.ai

The largest catalog of offers by far. If you want to compare GPUs or find obscure hardware, Vast.ai probably has it. The marketplace model means you'll see everything from consumer RTX cards to datacenter A100s and H100s, often at competitive prices.

The trade-off is stability. It's a marketplace — you're renting from individual hosts, and quality varies. It's common to land on machines with outdated drivers, flaky networking, or other issues that force you to destroy and re-provision. For long-running jobs, this is a problem. For quick experiments where you can tolerate the occasional bad host, it's fine.

To understand why: when Skyward provisions an instance, it connects via SSH, runs a bootstrap script (install Python, dependencies, set up the worker), and then starts streaming events. On AWS or GCP, this pipeline almost never fails — the infrastructure is consistent. On marketplace providers like Vast.ai, any step can fail: the SSH port might not be reachable, the system packages might be outdated, a driver might be incompatible. When that happens, Skyward detects the failure and you need to re-provision, which means starting over with a different host.

I use Vast.ai when I want to compare performance across different GPUs for a specific workload, or when I need a GPU type that isn't available on my usual providers. For multi-node setups, Vast.ai uses overlay networks to allow inter-node communication — it works, but adds a layer of complexity compared to VPC-native networking.

### Hyperstack

Excellent pricing for A100s and H100s — genuinely some of the best on-demand rates I've seen for those GPUs. A100 at $1.35/hr and H100 at $1.90/hr on-demand. If your workload needs datacenter-grade hardware, Hyperstack is worth checking.

The catalog is narrower than RunPod or Vast.ai, but the price-to-performance ratio for high-end GPUs is strong.

### Verda

The best spot pricing for H100s I've found — $0.80/hr for a single H100. That's significantly cheaper than AWS or GCP spot for the same hardware.

The catch: availability is inconsistent. GPUs go out of stock frequently. When they're available, the value is excellent. When they're not, you need a fallback. This is exactly the scenario multi-spec selection was designed for:

```python
sky.Compute(
    sky.Spec(provider=sky.Verda(), accelerator=sky.accelerators.H100()),
    sky.Spec(provider=sky.AWS(), accelerator=sky.accelerators.H100()),
    selection="cheapest",
)
```

Skyward ranks all offers from both providers by price into a single list. Verda's H100 at $0.80/hr will rank higher than AWS's. If Verda has availability, you get the cheap price. If it doesn't, the provisioning fails for that offer, Skyward moves to the next one in the ranked list, and you land on AWS transparently. You don't need to handle the fallback yourself — the pool actor tries each offer in the chain until one succeeds.

### TensorDock

Competitive pricing and a wide geographic spread (100+ locations). I've had some stability issues — similar to Vast.ai, the quality of the underlying hardware varies. I use it less frequently than RunPod or AWS, but it's a reasonable option for cost-conscious experimentation where you can tolerate occasional re-provisioning.

One quirk: TensorDock maps internal ports to random external ports. SSH is never on port 22 externally. Skyward handles this automatically, but if you're debugging connectivity issues, this is worth knowing.

### GCP

Slower to provision than AWS, and the setup (Application Default Credentials, project IDs, quota requests) is more involved. The distinguishing factor is TPU access — if your workload benefits from TPUs, GCP is the only game in town.

For GPU workloads specifically, I don't reach for GCP over AWS unless there's a specific reason. Provisioning is noticeably slower, and the quota system means you often need to request access before you can use a specific GPU in a specific zone.

### JarvisLabs

Extremely fast provisioning — instances come up almost instantly. The reason: JarvisLabs uses pre-built Docker images with frameworks already installed. Most providers start from a bare OS image and run a full bootstrap (install Python, create a venv, install dependencies, set up the worker) — that takes minutes. JarvisLabs skips most of that because the image already has what you need. The trade-off is a more limited catalog and less flexibility in the base image.

Good for quick iterations where provisioning latency matters more than hardware breadth.

### Scaleway, Vultr

Both work well with Skyward. I don't have strong opinions or specific experiences to share beyond that. They're solid options if their regions or pricing fit your needs.

## Saving money

### Spot instances everywhere

For any workload that can tolerate interruption — and most experimentation can — spot instances are the single biggest cost lever. The savings range from 50% to 90% depending on provider and GPU type.

The key insight with spot is that **a small bid premium prevents most preemptions**. On RunPod, a bid multiplier of 1.2-1.4x makes spot instances behave almost like on-demand in practice. You're paying 20-40% more than the minimum spot price, but still far less than on-demand.

Spot mechanics vary by provider. AWS uses EC2 Fleet with a capacity-optimized allocation strategy — it picks instance pools where interruption is least likely. RunPod and Vast.ai use bid-based pricing where the multiplier directly controls how much you're willing to pay above the minimum. The strategies are different, but the principle is the same: a modest premium buys significantly more stability.

If a spot instance does get preempted, Skyward detects it automatically. The node actor notices the instance is gone, notifies the pool, and the reconciler provisions a replacement using the same offer. Your code doesn't need to handle this — but your workflow should be resilient to restarts.

### Checkpoints with volumes

The complement to spot instances: **save intermediate work to persistent volumes**. If your instance gets preempted, you re-provision and resume from the last checkpoint instead of starting over. This turns spot interruption from a catastrophe into a minor inconvenience.

```python
sky.Compute(
    provider=sky.RunPod(bid_multiplier=1.3),
    accelerator=sky.accelerators.RTX_4090(),
    allocation="spot",
    volume=sky.Volume(name="training-checkpoints", mount="/data"),
)
```

### Pick the right GPU, not the biggest

Don't default to A100s because they're the "standard." For many workloads, an RTX 3090 at a fraction of the cost will give you equivalent throughput. The only way to know is to test your specific pipeline on different hardware and compare.

### Choose the right provider for the job

I don't use one provider for everything, and neither should you. My rule of thumb:

- **Quick experiment, cost-sensitive**: RunPod Community Cloud with RTX 3090/4090 spot
- **Long training run, can't afford failures**: AWS with spot-if-available
- **H100 on a budget**: Verda spot (with AWS fallback via multi-spec)
- **A100/H100 on-demand, best price**: Hyperstack
- **Comparing GPUs across tiers**: Vast.ai (largest catalog)
- **Fastest possible iteration cycle**: JarvisLabs (near-instant provisioning)

## For beginners

If you're just getting started with GPU cloud, here's my advice:

**Start with RunPod.** It has the lowest barrier to entry — one API key, no IAM, no VPC. You can go from zero to a running GPU in minutes.

**Test different GPUs on your own workload.** Don't pick a GPU based on specs or other people's benchmarks. Rent an RTX 3090 for an hour, run your pipeline, measure throughput. Then try an RTX 4090. Then an A100. The best cost-performance ratio depends entirely on your specific workload — memory bandwidth, compute density, batch size, model architecture all affect which GPU wins.

**Use spot from the start.** Get comfortable with spot instances and checkpoint-based workflows early. The cost savings are too significant to leave on the table, and the discipline of writing checkpoint-resumable code will serve you well regardless.

**Don't fear re-provisioning.** On smaller providers, instances sometimes fail during bootstrap — a port isn't reachable, a driver is outdated, something in the host environment is off. This is normal. Destroy and re-provision. It's not a bug in your code. As you get more comfortable, you'll gravitate toward more stable providers (AWS, GCP) for workloads where this matters.

---

## Related topics

- **[Providers](providers.md)** — technical reference with setup, parameters, and IAM permissions
- **[Accelerators](accelerators.md)** — GPU specifications and selection
- **[Multi-provider selection](guides/multi-provider.md)** — using multiple providers with fallback
- **[Compare accelerators](compare.md)** — interactive GPU comparison
