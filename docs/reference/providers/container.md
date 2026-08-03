# Container

`sky.Container` creates an account descriptor for the local `container` provider kind. It needs no credentials and uses the selected local container runtime to create nodes.

```python
import skyward as sky

provider = sky.Container(binary="docker", name="local")
```

::: skyward.Container
