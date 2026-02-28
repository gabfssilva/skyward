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
- Lambda: Lambda Cloud (bare-metal GPUs, on-demand)
- RunPod: GPU cloud (pods, serverless endpoints)
- VastAI: GPU marketplace (Docker containers, spot/bid pricing)
- Thunder: Thunder Compute (GPU cloud, Quebec, per-minute billing)
- Verda: GPU cloud (dedicated instances, spot pricing)
- Hyperstack: GPU cloud (bare-metal, environment-scoped resources)

NOTE: Only config classes are imported at module level to avoid pulling in
SDK dependencies (aioboto3, google-cloud-compute, etc.).
"""

# Only import config classes - these have NO SDK dependencies
# This allows `import skyward as sky` without requiring provider SDKs
from .aws.config import AWS
from .container.config import Container
from .gcp.config import GCP
from .hyperstack.config import Hyperstack
from .lambda_cloud.config import Lambda
from .runpod.config import RunPod
from .thunder.config import ThunderCompute
from .vastai.config import VastAI
from .verda.config import Verda

__all__ = [
    # Config classes only - handlers must be imported explicitly
    "AWS",
    "Container",
    "GCP",
    "Hyperstack",
    "Lambda",
    "RunPod",
    "ThunderCompute",
    "VastAI",
    "Verda",
]
