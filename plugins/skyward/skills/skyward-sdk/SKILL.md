---
name: skyward-sdk
description: Write Python against the Skyward SDK — @sky.function, Compute/Spec, the
  dispatch operators (>> @ & >), accelerators and providers, distributed training and
  collections, plugins (torch/jax/keras/cuml), volumes, events, notebook kernel. The
  reference is the docs site; fetch the page for the task instead of recalling the API.
  Triggers on skyward, @sky.function, sky.Compute, sky.shard, skyward plugin,
  skyward volume, distributed training on skyward.
---

# Using Skyward (Python)

Don't write Skyward code from memory — the API moves. Fetch the page for what you're
doing, read it, then write. One page is usually enough; don't pull the whole map.

Base: `https://gabfssilva.github.io/skyward/`

## Where to go

| you're doing | page |
|---|---|
| install, first remote job | `getting-started/`, `guides/hello-skyward/` |
| lazy functions, operators, Computes, tasks, leases | `concepts/` |
| `Compute`, `Spec`, `Pending`, `Group`, `Streaming` | `reference/pool/` |
| fan out, gather, broadcast, shard, stream results | `guides/parallel-execution/`, `guides/broadcast/`, `guides/data-sharding/`, `guides/streaming/` |
| code running *on* the node: `instance_info`, `shard`, output | `reference/runtime/` |
| picking a GPU, catalog, prices | `accelerators/`, `compare/`, `guides/using-accelerators/` |
| provider accounts, cheapest across clouds | `providers/`, `guides/multi-provider/`, `choosing-a-provider/` |
| one provider's options (aws, gcp, runpod, vastai, lambda, …) | `reference/providers/<name>/` |
| multi-node training | `distributed-training/`, `guides/pytorch-distributed/` |
| HuggingFace, FSDP, Keras training | `guides/huggingface-finetuning/`, `guides/fsdp-huggingface/`, `guides/keras-training/` |
| shared state: dict, set, counter, queue, barrier, lock | `distributed-collections/`, `reference/distributed/` |
| framework plugins | `plugins/`, `plugins/<torch|jax|keras|accelerate|cuml|joblib|sklearn|mps|mig>/` |
| S3/GCS as a filesystem, getting artifacts back | `volumes/`, `guides/s3-volumes/`, `guides/torch-model-roundtrip/` |
| splitting one GPU | `guides/fractional-gpus/`, `guides/nvidia-mig/` |
| watching progress, events, callbacks | `callbacks/`, `reference/events/` |
| control-plane URL, provider creds, Compute options | `reference/config/` |
| running it locally before paying for a GPU | `guides/local-containers/`, `guides/worker-executors/` |
| Jupyter against a remote kernel | `notebook/` |
| how the daemon works, HTTP API, persistence | `architecture/`, `http-api/`, `persistence/`, `provision-controllers/` |

Nothing matches, or a URL 404s: `llms.txt` is the full index. `llms-full.txt` is the whole
surface in one file (~11 KB) when you need breadth rather than a page.

The `sky` command line is the other skill: `using-skyward-cli`.

The site tracks `main`. If docs and installed package disagree, the package wins:
`sky version`, then `python -c "import skyward as sky; help(sky.Compute)"`.

No network: say so and stop. Don't guess the API.
