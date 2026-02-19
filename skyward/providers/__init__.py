"""Cloud providers for Skyward.

Each provider implements the Provider[C, S] protocol:
- prepare → provisions cluster-level infrastructure
- provision → launches instances
- get_instance → polls instance status
- terminate → destroys instances
- teardown → cleans up cluster resources

Available providers:
- AWS: Amazon Web Services (EC2 Fleet, spot instances)
- GCP: Google Cloud Platform (Compute Engine, instance templates, bulk_insert)
- RunPod: GPU cloud (pods, serverless endpoints)
- VastAI: GPU marketplace (Docker containers, spot/bid pricing)
- Verda: GPU cloud (dedicated instances, spot pricing)

NOTE: Only config classes are imported at module level to avoid pulling in
SDK dependencies (aioboto3, google-cloud-compute, etc.).
"""

# Only import config classes - these have NO SDK dependencies
# This allows `import skyward as sky` without requiring provider SDKs
from .aws.config import AWS
from .container.config import Container
from .gcp.config import GCP
from .runpod.config import RunPod
from .vastai.config import VastAI
from .verda.config import Verda

__all__ = [
    # Config classes only - handlers must be imported explicitly
    "AWS",
    "Container",
    "GCP",
    "RunPod",
    "VastAI",
    "Verda",
]
