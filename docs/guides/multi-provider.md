# Multi-provider selection

Accelerator prices fluctuate. Availability varies by region, time of day, and provider. An A100 on VastAI might cost $1.50/hr right now but have no capacity, while the same accelerator on AWS is $3.00/hr with instant availability. Instead of checking pricing pages manually, you can describe your hardware need across multiple providers and let Skyward query all of them, compare offers, and provision from the best option.

## The spec

A `sky.Spec` bundles a provider with hardware preferences into a single, composable unit:

```python
sky.Spec(
    provider=sky.VastAI(),
    accelerator="A100",
    max_hourly_cost=2.50,
    allocation="spot",
)
```

It carries the same fields you'd normally pass to `ComputePool` — `accelerator`, `nodes`, `vcpus`, `memory_gb`, `architecture`, `allocation`, `region`, `max_hourly_cost`, `ttl` — but scoped to a specific provider. This separation is what makes cross-provider comparison possible: each `Spec` is a self-contained description of "what I want, from whom."

## Cheapest across providers

Pass multiple `Spec` objects to `ComputePool` and Skyward queries each provider's available offers, compares prices, and provisions from the cheapest:

```python
--8<-- "examples/guides/12_multi_provider.py:26:33"
```

With `selection="cheapest"` (the default), Skyward calls `offers()` on every provider in the list, collects all available machine types with their pricing, and picks the one with the lowest cost. The pool then provisions from that provider — the rest of the lifecycle (SSH, bootstrap, task dispatch) is unchanged.

This is useful when you don't have a strong provider preference and want cost optimization. The same A100 workload might end up on VastAI one day and AWS the next, depending on current market prices.

## First available

When you have a preferred provider but want a fallback, use `selection="first"`:

```python
--8<-- "examples/guides/12_multi_provider.py:36:43"
```

Specs are tried in order. Skyward queries RunPod first — if it has H100 offers available, that's where the pool provisions. If RunPod has no capacity, Skyward moves to AWS. This gives you deterministic priority ordering while still avoiding manual retries when your preferred provider is out of stock.

## Per-spec constraints

Each `Spec` can have its own allocation strategy, cost cap, and region. This enables escalating fallback patterns — start aggressive, fall back to safer options:

```python
--8<-- "examples/guides/12_multi_provider.py:46:66"
```

The first spec tries spot instances on VastAI with a $2.50/hr cap — the cheapest option if available. If that fails (no capacity, or prices exceed the cap), the second spec tries Verda with spot-if-available. The third spec is the safety net: on-demand AWS, which is more expensive but virtually always available. Skyward evaluates all three and picks the cheapest viable option.

## How it works

When `ComputePool.__enter__` runs, Skyward iterates through your specs before provisioning anything:

1. For each `Spec`, it creates a provider instance and calls `provider.offers(spec)` — an async generator that yields available machine types with pricing.
2. Based on the `selection` strategy, it either takes the first available offer or collects all offers and picks the cheapest.
3. The winning offer — which carries the selected provider, machine type, and pricing — is passed to `provider.prepare(spec, offer)` to set up infrastructure.
4. From there, the lifecycle proceeds normally: provision instances, bootstrap, start workers.

The key insight is that offer querying is fast (API calls to check availability and pricing) while provisioning is slow (launching machines). By querying all providers before committing to one, Skyward makes an informed decision without wasting time on failed provisioning attempts.

## Run the full example

```bash
git clone https://github.com/gabfssilva/skyward.git
cd skyward
uv run python examples/guides/12_multi_provider.py
```

---

**What you learned:**

- **`sky.Spec`** bundles a provider with hardware preferences into a composable unit — accelerator, nodes, allocation, cost cap, all scoped to one provider.
- **Multi-spec `ComputePool`** accepts multiple Specs as positional arguments and selects the best offer before provisioning.
- **`selection="cheapest"`** (default) queries all providers and picks the lowest price across all of them.
- **`selection="first"`** respects your priority ordering — tries specs in sequence, stops at the first with available offers.
- **Per-spec constraints** let each Spec have its own allocation strategy and cost cap, enabling escalating fallback patterns.
- **Single-provider mode still works** — `ComputePool(provider=sky.AWS(), ...)` is unchanged and internally wraps a single `Spec`.
