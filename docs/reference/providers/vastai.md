# VastAI

`sky.VastAI` creates an account descriptor for the `vastai` provider kind. The factory exposes marketplace reliability, CUDA, geography, price, network, image, disk, and request filters.

```python
import skyward as sky

provider = sky.VastAI(name="marketplace", min_reliability=0.95, geolocation="US")
```

::: skyward.VastAI
