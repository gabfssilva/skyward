# Compare accelerators

The widget compares a curated set of hardware specifications. It is a static
comparison view, not the source of truth for request names, provider
availability, or current prices.

The canonical request catalog is documented in [Accelerators](accelerators.md).
Use the daemon's offer cache for live provider data:

```bash
sky offers list --accelerator H100 --limit 10
sky offers summary --accelerator A100
```

Select up to eight entries below.

<div id="gpu-compare">
  <div class="gpu-compare-slots"></div>
  <div class="gpu-compare-result"></div>
  <div class="gpu-compare-charts"></div>
</div>
