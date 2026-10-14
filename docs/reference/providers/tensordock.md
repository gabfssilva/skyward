# TensorDock

`sky.TensorDock` creates an account descriptor for the `tensordock` provider kind. Location, tier, storage, operating system, and request settings constrain the catalog and launch. The VM's vCPUs and RAM are the compute's `cpus` and `memory_gb`.

```python
import skyward as sky

provider = sky.TensorDock(name="production", location="United States")
```

::: skyward.TensorDock
