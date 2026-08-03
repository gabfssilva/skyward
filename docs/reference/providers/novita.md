# Novita

`sky.Novita` creates an account descriptor for the `novita` provider kind. Cluster, image, CUDA, root filesystem, and request settings are account configuration.

```python
import skyward as sky

provider = sky.Novita(name="production", min_cuda_version="12.8")
```

::: skyward.Novita
