# GCP

`sky.GCP` is the account descriptor for the `gcp` provider kind. `service_account_json` accepts the JSON contents; `GOOGLE_APPLICATION_CREDENTIALS` names a file whose contents are read in the client process.

The adapter needs `skyward[gcp]`, which brings cryptography for the service-account signature. Without it the daemon does not register the kind.

```python
import skyward as sky

provider = sky.GCP(project="training", zone="us-central1-a", name="production")
```

::: skyward.GCP
