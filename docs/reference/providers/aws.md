# AWS

`sky.AWS` creates an account descriptor for the `aws` provider kind. Credentials are read from explicit arguments or the AWS environment/profile. Use `name` to keep multiple AWS accounts in one daemon.

The adapter needs `skyward[aws]`, which brings aioboto3. Without it the daemon does not register the kind.

```python
import skyward as sky

provider = sky.AWS(name="production", region="us-east-1")
```

::: skyward.AWS
