# Vultr

`sky.Vultr` creates an account descriptor for the `vultr` provider kind. `mode` selects the cloud or bare-metal catalog; `region` and `os_id` configure placement and the operating system.

```python
import skyward as sky

provider = sky.Vultr(name="production", mode="cloud", region="ewr")
```

::: skyward.Vultr
