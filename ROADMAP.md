# Roadmap

Skyward is in active feature exploration. We're testing different APIs, providers, tools, and plugins — some ideas prove their worth after real-world use, others get dropped. The public API is not yet stable, and that's intentional: we'd rather ship the right abstractions than freeze the wrong ones.

Contributions are welcome on anything here.

## More ways to use Skyward

### CLI

Today, all interaction with Skyward happens through Python. A `sky` command-line tool would open the door to quick exploration, pool management, and script execution without writing a single line of code.

- **Browse and discover** — search GPU offers across providers, compare pricing, and filter by accelerator type, region, or budget
- **Manage pools** — create, inspect, resize, and destroy pools from the terminal
- **Connect** — SSH into running instances and stream logs in real time
- **Run remotely** — `sky run script.py` ships a script to a pool and streams output back

```bash
sky offers --accelerator A100 --max-price 2.50
sky pool create --provider aws --accelerator A100 --nodes 4
sky ssh my-pool
sky run train.py --pool my-pool
sky logs my-pool --follow
```

### Notebooks

Provisioning a cloud GPU should be as easy as opening a notebook. We want first-class support for interactive environments running on Skyward-managed instances.

- **Marimo** — `sky notebook` provisions a marimo server on a pool instance and tunnels it to your browser
- **Jupyter** — same flow for Jupyter, with proper kernel management and persistent storage
- **VS Code Remote** — connect VS Code directly to pool instances via SSH, with extensions and workspace sync

## More clouds

Each new provider expands where Skyward can find capacity and competitive pricing. The provider protocol is well-established — adding a new cloud means implementing `offers`, `prepare`, `provision`, `get_instance`, `terminate`, and `teardown`.

- **Vultr** — bare-metal GPU instances with hourly billing, A100/H100/L40S/B200 availability
- **Novita AI** — inference-optimized GPU cloud with pay-per-use pricing and fast cold starts
- **CoreWeave** — Kubernetes-native GPU cloud with strong H100/B200 availability
- **FluidStack** — aggregated GPU capacity from multiple data centers, competitive spot pricing
- **Paperspace** — Gradient ecosystem, good for notebook-first workflows
- **OCI (Oracle Cloud)** — enterprise GPU instances, bare-metal A100/H100 clusters with RDMA

Contributions here have high impact — each provider unlocks new hardware and pricing for every Skyward user.

## Smarter orchestration

### Better preemption recovery

Spot instances get interrupted. Today Skyward detects and replaces preempted nodes, but there's room to make this faster and more seamless — quicker detection across providers, automatic task retry for in-flight work, and state preservation where possible.

### Multi-provider intelligence

Beyond the current `"cheapest"` selection strategy: automatic failover when a provider's quota is exhausted, cross-provider cost optimization that factors in real-time availability, and smarter spec ordering based on historical reliability.

### Heterogeneous pools

Currently, all nodes in a pool share the same spec. Heterogeneous pools would allow mixing node types — for example, A100s for training and T4s for preprocessing within the same pool. This opens the door to cost-optimized pipelines where each stage runs on the right hardware.

```python
with sky.Compute(
    sky.Spec(provider=sky.AWS(), accelerator="A100", nodes=2),
    sky.Spec(provider=sky.AWS(), accelerator="T4", nodes=4),
) as pool:
    preprocessed = preprocess(raw) >> pool  # routes to T4s
    result = train(preprocessed) >> pool    # routes to A100s
```

The API for routing tasks to specific node types is still an open question.

## Safer defaults

### SSH host key verification

Today Skyward uses `known_hosts=None`, which disables host key verification. This is convenient but insecure. Proper key management — storing and verifying host keys per provider and instance — would close this gap without hurting the user experience.

### Serialization safety

Skyward uses cloudpickle for function serialization. Deserializing untrusted payloads is a known risk. We want guardrails: allowlisting safe types, sandboxing deserialization, and clear warnings when running code from untrusted sources.

### Secrets handling

API keys, tokens, and credentials should never appear in bootstrap scripts or logs. A secrets injection mechanism — environment variables forwarded over SSH, or integration with secret managers — would keep credentials out of observable surfaces.

### Network isolation

Default firewall rules and security groups per provider, so new pools aren't wide open by default. Provider-specific implementations (AWS security groups, GCP firewall rules, etc.) with sensible defaults that users can override.

## More visibility

### Observability

Prometheus metrics and OpenTelemetry traces for pool lifecycle, task execution, node health, and provider API calls. This is the foundation for dashboards, alerting, and debugging distributed workloads in production.

### Actionable error messages

Clear diagnostics for the most common failure modes: bootstrap failures (what command failed and why), SSH timeouts (is it a security group? a key issue?), and serialization errors (what object couldn't be pickled and why). Error messages should suggest next steps, not just report what went wrong.

## More integrations

Plugins compose with `@sky.function` to set up frameworks, inject credentials, and manage distributed state. Each plugin below would follow the existing plugin protocol — `transform`, `bootstrap`, `decorate`, and lifecycle hooks.

- **MLflow** — automatic experiment tracking, artifact logging, and model registry integration across distributed runs
- **Weights & Biases** — run initialization, metric logging, and artifact sync per node
- **DeepSpeed** — ZeRO optimizer stages, model parallelism configuration, and launcher integration for multi-node training
- **vLLM** — deploy inference servers on pool instances with automatic tensor parallelism across available GPUs
- **Axolotl** — fine-tuning configuration management, dataset preparation, and distributed training orchestration

## Docs & community

- **End-to-end examples** — complete walkthroughs for common workflows: fine-tuning an LLM, serving inference at scale, distributed training with PyTorch, multi-provider cost optimization
- **Contributing guide** — how to set up the dev environment, run tests, add a provider, write a plugin, and submit changes
