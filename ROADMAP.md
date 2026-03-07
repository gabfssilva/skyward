# Roadmap

Where Skyward is headed. No deadlines, no priority order — just direction.
Contributions are welcome on anything here.

## CLI

- [ ] `sky` command-line interface
  - [ ] Browse offers and list providers
  - [ ] Manage pools, SSH into instances, stream logs
  - [ ] Run scripts remotely with `sky run`

## Notebook Support

- [ ] Marimo — `sky notebook` provisions a marimo server on a pool instance
- [ ] Jupyter — provision and connect to Jupyter on pool instances
- [ ] VS Code Remote — connect VS Code directly to pool instances via SSH

## Safety

- [ ] SSH host key verification — proper key management instead of `known_hosts=None`
- [ ] Serialization safety — protect against untrusted deserialization
- [ ] Secrets handling — keep credentials out of bootstrap scripts and logs
- [ ] Network isolation — default firewall rules and security groups per provider

## Providers

- [ ] Lambda Cloud
- [ ] CoreWeave
- [ ] FluidStack
- [ ] Paperspace
- [ ] OCI
## Core

- [ ] Better preemption recovery — faster detection and replacement across providers
- [ ] Multi-provider intelligence — automatic fallback and cross-provider cost optimization

## Observability

- [ ] Prometheus and OpenTelemetry support

## Developer Experience

- [ ] Actionable error messages — clear diagnostics for bootstrap failures, SSH timeouts, and serialization errors

## Ecosystem

- [ ] MLflow plugin
- [ ] Weights & Biases plugin
- [ ] DeepSpeed plugin
- [ ] vLLM plugin
- [ ] Axolotl plugin

## Docs & Community

- [ ] End-to-end examples — fine-tuning, inference, distributed training
- [ ] Contributing guide
