# Scaleway

`sky.Scaleway` creates an account descriptor for the `scaleway` provider kind. Pass the project credentials explicitly or use `SCW_SECRET_KEY` and `SCW_DEFAULT_PROJECT_ID`.

```python
import skyward as sky

provider = sky.Scaleway(name="production", zone="fr-par-2")
```

::: skyward.Scaleway
