# VastAI

`sky.VastAI` is the account descriptor for the `vastai` provider kind. It exposes marketplace reliability, CUDA, geography, price, network, image, disk, and request filters.

```python
import skyward as sky

provider = sky.VastAI(name="marketplace", min_reliability=0.95, geolocation="US")
```

::: skyward.VastAI
