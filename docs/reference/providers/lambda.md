# Lambda

`sky.Lambda` creates an account descriptor for the `lambda` provider kind. The optional `region` selects a preferred location; the adapter can choose available capacity when it is omitted.

```python
import skyward as sky

provider = sky.Lambda(name="production", region="us-east-3")
```

::: skyward.Lambda
