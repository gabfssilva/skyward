# JarvisLabs

`sky.JarvisLabs` creates an account descriptor for the `jarvislabs` provider kind. Use `region`, `template`, and `storage_gb` to constrain the account's catalog and launches.

```python
import skyward as sky

provider = sky.JarvisLabs(name="production", region="IN2")
```

::: skyward.JarvisLabs
