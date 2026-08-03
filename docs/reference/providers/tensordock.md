# TensorDock

`sky.TensorDock` creates an account descriptor for the `tensordock` provider kind. Location, tier, storage, operating system, resource minima, and request settings constrain the catalog and launch.

```python
import skyward as sky

provider = sky.TensorDock(name="production", location="United States")
```

::: skyward.TensorDock
