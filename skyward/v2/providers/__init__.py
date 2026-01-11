"""Cloud providers for Skyward v2.

Each provider is an event-driven component that handles:
- ClusterRequested → provisions infrastructure
- InstanceRequested → launches instances
- ShutdownRequested → terminates resources

Available providers:
- AWS: Amazon Web Services (EC2 Fleet, spot instances)
- VastAI: GPU marketplace (Docker containers, spot/bid pricing)
- Verda: GPU cloud (dedicated instances, spot pricing)

NOTE: Only config classes are imported at module level to avoid pulling in
SDK dependencies (aioboto3, httpx, etc.). Handlers and modules should be
imported explicitly when needed:

    from skyward.v2.providers.aws import AWSHandler, AWSModule
"""

# Only import config classes - these have NO SDK dependencies
# This allows `import skyward.v2 as sky` without requiring provider SDKs
from .aws.config import AWS
from .vastai.config import VastAI
from .verda.config import Verda

__all__ = [
    # Config classes only - handlers must be imported explicitly
    "AWS",
    "VastAI",
    "Verda",
]
