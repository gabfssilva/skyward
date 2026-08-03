# RunPod

`sky.RunPod` creates an account descriptor for the `runpod` provider kind. Its configuration controls the image, storage, location filters, market settings, networking, and request behavior.

```python
import skyward as sky

provider = sky.RunPod(name="production", cloud_type="secure")
```

::: skyward.RunPod
