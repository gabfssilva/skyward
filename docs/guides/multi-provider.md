# Multi-provider selection

Availability and price vary by provider. A `Compute` can receive several `Spec` values and choose one matching offer before it provisions the nodes.

## The spec

A `sky.Spec` binds a provider to machine requirements:

```python
sky.Spec(
    provider=sky.VastAI(),
    accelerator=sky.accelerators.A100(),
    max_hourly_cost=2.50,
)
```

`Spec` contains `provider`, `accelerator`, `cpus`, `memory_gb`, `region`, `disk_gb`, `architecture`, and `max_hourly_cost`. Node count, allocation, selection, image, plugins, volumes, and task options belong to `Compute`, because they apply to the selected specification rather than to one provider alternative.

## Cheapest across providers

Pass multiple `Spec` objects to `Compute`. With `selection="cheapest"`, the control plane compares matching cached offers and selects the lowest priced viable option:

```python
--8<-- "guides/12_multi_provider.py:26:34"
```

The provider can change between runs as its offer cache changes. The compute lifecycle after selection is the same: provision, bootstrap, start workers, and dispatch tasks.

## First available

Use `selection="first"` when the order of the specifications is the priority order:

```python
--8<-- "guides/12_multi_provider.py:37:46"
```

The selected `Spec` determines the provider and machine shape. `nodes=4` and `allocation="on_demand"` apply to the compute as a whole.

## Per-provider constraints

Constraints that identify a provider alternative stay on its `Spec`, while shared compute settings remain on `Compute`:

```python
--8<-- "guides/12_multi_provider.py:49:66"
```

Here the VastAI alternative has a cost cap, while the Verda and AWS alternatives are fallback shapes. The allocation policy is shared by the compute; it is not a field of `Spec`.

## Single-provider mode

For one provider, use the direct `Compute` form:

```python
with sky.Compute(
    provider=sky.AWS(),
    accelerator=sky.accelerators.A100(),
    nodes=2,
    allocation="spot_if_available",
) as compute:
    train(10) >> compute
```

This form is equivalent to a compute with one `Spec`.

## Run the full example

```bash
git clone https://github.com/gabfssilva/skyward.git
cd skyward
uv run python guides/12_multi_provider.py
```

---

**What you learned:**

- **`sky.Spec`** binds a provider to hardware and offer constraints.
- **Multi-spec `Compute`** chooses one provider alternative before provisioning.
- **`selection="cheapest"`** compares viable matching offers.
- **`selection="first"`** uses specification order as priority.
- **`allocation`, `nodes`, and `image`** are compute-level settings shared by the selected specification.
