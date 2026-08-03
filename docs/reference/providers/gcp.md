# GCP

`sky.GCP` creates an account descriptor for the `gcp` provider kind. `service_account_json` accepts the JSON contents; `GOOGLE_APPLICATION_CREDENTIALS` names a file whose contents are loaded by the factory.

The adapter needs `skyward[gcp]`, which brings cryptography for the service-account signature. Without it the daemon does not register the kind.

```python
import skyward as sky

provider = sky.GCP(project="training", zone="us-central1-a", name="production")
```

::: skyward.GCP
