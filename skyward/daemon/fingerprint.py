"""Deterministic pool fingerprinting for inline daemon mode.

Derives a human-readable slug from identity-defining fields
(provider, accelerator, region, image) so inline daemon pools
can be matched across script runs.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skyward.api.spec import Spec


def compute_fingerprint(spec: Spec) -> str:
    """Compute a deterministic fingerprint for a spec.

    Identity fields: provider type, accelerator, region, image.
    Operational fields (nodes, ttl, allocation, timeouts) are ignored.

    Parameters
    ----------
    spec
        The user-facing Spec to fingerprint.

    Returns
    -------
    str
        Slug like ``aws-A100-us-east-1-a1b2c3``.
    """
    provider_type = spec.provider.type
    accel_name = spec.accelerator.name if spec.accelerator else "cpu"
    region = spec.region or "default"
    image_hash = spec.image.content_hash()

    content = json.dumps(
        {
            "provider": provider_type,
            "accelerator": accel_name,
            "region": region,
            "image": image_hash,
        },
        sort_keys=True,
    )
    short_hash = hashlib.sha256(content.encode()).hexdigest()[:6]
    return f"{provider_type}-{accel_name}-{region}-{short_hash}"
