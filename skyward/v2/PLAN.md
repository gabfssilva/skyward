# Skyward v2 - Status Completo e Próximos Passos

## Arquitetura v2

**Princípios:**
- 100% event-driven com asyncio
- DI extensivo via `injector` library
- Eventos via `blinker` (native async)
- Mais OOP que funcional (service classes, @component)
- Imutabilidade para configs/events, mutabilidade para state

**Padrões:**
- `@component`: Auto-gera `__init__`, aplica `@inject`, registra handlers
- `@on(EventType)`: Marca métodos como event handlers
- `@monitor(interval)`: Background loops com DI
- `Client[T]`: Factory que retorna async context manager (DI pattern)

---

## STATUS: O Que Já Foi Feito

### Core (100% Completo)

| Arquivo | Status | Descrição |
|---------|--------|-----------|
| `events.py` | ✅ | 20+ event types (Requests + Facts), type aliases, InstanceInfo |
| `bus.py` | ✅ | AsyncEventBus com emit/emit_await/request, usa blinker send_async |
| `app.py` | ✅ | @component, @on, @monitor, create_app, app_context, MonitorManager |
| `spec.py` | ✅ | PoolSpec, ImageSpec, AllocationStrategy |
| `protocols.py` | ✅ | Transport, Executor, TransportFactory, HealthChecker, PreemptionChecker |
| `node.py` | ✅ | Node component com state machine (INIT→PROVISIONING→BOOTSTRAPPING→READY→REPLACING) |
| `pool.py` | 🟡 | ComputePool component, start/stop funcionam, **run/broadcast são stubs** |

### AWS Provider (80% Completo)

| Arquivo | Status | Descrição |
|---------|--------|-----------|
| `providers/aws/config.py` | ✅ | AWS dataclass imutável |
| `providers/aws/state.py` | ✅ | AWSResources, AWSClusterState, InstanceConfig |
| `providers/aws/clients.py` | ✅ | `Client[T]` type, AWSModule com providers para EC2/S3/IAM/STS |
| `providers/aws/handler.py` | 🟡 | @on handlers para Cluster/Instance/Shutdown, **veja pendências abaixo** |

**Pendências no AWSHandler:**
1. `_resolve_instance_config()` - Mapping accelerator→instance_type **hardcoded**
2. `_get_dlami()` - AMI **hardcoded** por região
3. `_generate_user_data()` - Bash básico, **deveria usar bootstrap DSL**
4. `_wait_bootstrap()` - **STUB: apenas sleep(10)**, deveria poll SSH

### Transport (100% Completo)

| Arquivo | Status | Descrição |
|---------|--------|-----------|
| `transport/ssh.py` | ✅ | SSHTransport completo: run, run_stream, upload, download, file ops |
| `transport/__init__.py` | ✅ | Exports |

### Bootstrap (100% Completo)

| Arquivo | Status | Descrição |
|---------|--------|-----------|
| `bootstrap/__init__.py` | ✅ | Re-export de skyward.bootstrap (DSL maduro do v1) |

### Monitors (50% Completo)

| Arquivo | Status | Descrição |
|---------|--------|-----------|
| `monitors.py` | 🟡 | InstanceRegistry ✅, MonitorModule ✅, **preemption/health são stubs** |

**Pendências:**
- `_check_instance_preemption()` - **STUB: sempre retorna False**
- `_ping_instance()` - **STUB: sempre retorna True**
- `check_aws_spot_interruption()` - ✅ Implementado (usa EC2 API)

---

## STATUS: O Que Falta Fazer

### Fase 7: Completar AWS Provider

#### 7.1 Bootstrap Polling via SSH
```python
# handler.py - _wait_bootstrap() atual:
async def _wait_bootstrap(self, info: InstanceInfo) -> None:
    await asyncio.sleep(10)  # STUB!

# Deveria:
async def _wait_bootstrap(self, info: InstanceInfo, timeout: float = 600) -> None:
    transport = SSHTransport(host=info.ip, user="ubuntu", key_path=self._ssh_key)
    async with transport:
        # Poll for bootstrap completion marker
        if await transport.wait_for_file("/tmp/bootstrap_complete", timeout=timeout):
            return
        raise TimeoutError("Bootstrap did not complete")
```

**Dependência:** Precisa de SSH key path no cluster state

#### 7.2 AMI Resolution via SSM
```python
# handler.py - _get_dlami() atual:
dlami_map = {"us-east-1": "ami-xxx", ...}  # HARDCODED!

# Deveria:
async def _get_dlami(self) -> str:
    async with self.ssm() as ssm:  # Novo client
        response = await ssm.get_parameter(
            Name="/aws/service/ecs/optimized-ami/amazon-linux-2/gpu/recommended"
        )
        return response["Parameter"]["Value"]["image_id"]
```

#### 7.3 Instance Type Mapping
```python
# handler.py - _resolve_instance_config() atual:
accelerator_map = {"T4": "g4dn.xlarge", ...}  # HARDCODED!

# Deveria: Query EC2 API ou tabela configurável
# Ou usar spec com instance_type explícito
```

#### 7.4 User Data com Bootstrap DSL
```python
# handler.py - _generate_user_data() atual:
lines = ["#!/bin/bash", f"export KEY={value}", ...]  # Básico!

# Deveria usar:
from skyward.v2.bootstrap import bootstrap, apt, pip, checkpoint
script = bootstrap(
    apt(*spec.image.apt),
    pip(*spec.image.pip),
    checkpoint("/tmp/bootstrap_complete"),
)
return resolve(script)
```

### Fase 8: Pool Execution

#### 8.1 Remote Function Execution
```python
# pool.py - run() atual:
async def run[T](self, fn, *args, node=None, **kwargs) -> T:
    raise NotImplementedError  # STUB!

# Implementação:
async def run[T](self, fn: Callable[..., T], *args, node: NodeId | None = None, **kwargs) -> T:
    target_node = self._nodes[node] if node else next(iter(self._nodes.values()))
    info = target_node.info

    # Create transport + executor
    transport = SSHTransport(host=info.ip, user="ubuntu", key_path=self._ssh_key)
    executor = RPyCExecutor(transport)

    async with transport:
        return await executor.execute(fn, *args, **kwargs)
```

#### 8.2 RPyC Executor
```python
# transport/rpyc.py (NOVO)
class RPyCExecutor:
    def __init__(self, transport: SSHTransport):
        self.transport = transport

    async def execute[T](self, fn: Callable[..., T], *args, **kwargs) -> T:
        # 1. Serialize with cloudpickle
        payload = cloudpickle.dumps((fn, args, kwargs))

        # 2. Send via SSH to RPyC server
        # 3. Receive and deserialize result
```

#### 8.3 Broadcast
```python
# pool.py - broadcast() atual:
async def broadcast[T](self, fn, *args, **kwargs) -> list[T]:
    raise NotImplementedError  # STUB!

# Implementação:
async def broadcast[T](self, fn: Callable[..., T], *args, **kwargs) -> list[T]:
    tasks = [
        self.run(fn, *args, node=node_id, **kwargs)
        for node_id in self._nodes
    ]
    return await asyncio.gather(*tasks)
```

### Fase 9: Monitors Completos

#### 9.1 Preemption Detection Genérico
```python
# monitors.py - _check_instance_preemption() atual:
async def _check_instance_preemption(info: InstanceInfo) -> tuple[bool, str | None]:
    return False, None  # STUB!

# Deveria: dispatch por provider
async def _check_instance_preemption(info: InstanceInfo) -> tuple[bool, str | None]:
    match info.provider:
        case "aws":
            return await check_aws_spot_interruption(info.id, region)
        case "digitalocean":
            return await check_do_interruption(info.id)
        case _:
            return False, None
```

#### 9.2 Health Check via SSH
```python
# monitors.py - _ping_instance() atual:
async def _ping_instance(info: InstanceInfo) -> bool:
    return True  # STUB!

# Deveria:
async def _ping_instance(info: InstanceInfo) -> bool:
    try:
        transport = SSHTransport(host=info.ip, user="ubuntu", key_path=KEY)
        async with asyncio.timeout(10):
            await transport.connect()
            code, _, _ = await transport.run("echo", "ping")
            return code == 0
    except Exception:
        return False
```

### Fase 10: Outros Providers

#### 10.1 DigitalOcean Provider ✅ COMPLETO
```
providers/digitalocean/
├── __init__.py    ✅ Exports
├── config.py      ✅ DigitalOcean dataclass
├── types.py       ✅ TypedDicts (DropletResponse, SizeResponse, etc)
├── client.py      ✅ DigitalOceanClient com @component, pydo.aio async
├── handler.py     ✅ @on handlers para Cluster/Instance/Shutdown
└── state.py       ✅ DOClusterState
```

#### 10.2 Vast.ai Provider ✅ COMPLETO
```
providers/vastai/
├── __init__.py    ✅ Exports
├── config.py      ✅ VastAI dataclass
├── types.py       ✅ TypedDicts (OfferResponse, InstanceResponse, etc)
├── client.py      ✅ VastAIClient com @component, httpx async
├── handler.py     ✅ @on handlers para Cluster/Instance/Shutdown
└── state.py       ✅ VastAIClusterState
```

#### 10.3 Verda Provider ✅ COMPLETO
```
providers/verda/
├── __init__.py    ✅ Exports
├── config.py      ✅ Verda dataclass
├── types.py       ✅ TypedDicts (InstanceTypeResponse, InstanceResponse, etc)
├── client.py      ✅ VerdaClient com @component, httpx async, OAuth2
├── handler.py     ✅ @on handlers para Cluster/Instance/Shutdown
└── state.py       ✅ VerdaClusterState
```

### Fase 11: Callbacks/Visualization (Futuro)

#### 11.1 Panel Callback (do v1)
- Visualização em tempo real
- Tracking de instâncias
- Métricas (CPU, GPU, memory)
- Logs agregados
- Cost tracking

#### 11.2 Approach v2
- Event handlers que escutam Metric, Log, TaskStarted, etc
- Rich/Panel para rendering
- Pode ser módulo separado: `skyward.v2.ui`

### Fase 12: Integrations (Futuro)

Do v1, precisamos portar:
- `integrations/torch.py` - Distributed setup
- `integrations/jax.py` - JAX setup
- `integrations/keras.py` - Keras utilities
- `integrations/joblib.py` - Parallel execution

### Fase 13: Data Utilities (Futuro)

Do v1:
- `cluster/utils.py` - InstanceInfo, instance_info()
- `cluster/sampler.py` - DistributedSampler, shard()

---

## Comparação v1 vs v2

| Feature | v1 | v2 | Status |
|---------|-----|-----|--------|
| Pool Management | ✅ | 🟡 | v2 falta run/broadcast |
| Instance Lifecycle | ✅ | ✅ | v2 async/event-driven |
| AWS Provider | ✅ | 🟡 | v2 falta bootstrap polling, AMI |
| DigitalOcean | ✅ | ✅ | pydo.aio async, TypedDicts, @component |
| Vast.ai | ✅ | ✅ | httpx async, TypedDicts, @component |
| Verda | ✅ | ✅ | httpx async, OAuth2, TypedDicts, @component |
| Bootstrap DSL | ✅ | ✅ | Reusado do v1 |
| Events | ~40 | ~20 | v2 mais focado |
| Callbacks/Panel | ✅ | ❌ | Não iniciado |
| Execution | ✅ | ❌ | Protocol definido, impl TBD |
| Torch/JAX/Keras | ✅ | ❌ | Não iniciado |
| Cost Tracking | ✅ | ❌ | Não iniciado |

---

## Prioridade de Implementação

### P0 - Crítico (Funcionalidade Básica)
1. [ ] `_wait_bootstrap()` - Poll SSH para bootstrap completion
2. [ ] `pool.run()` - Execute função remota
3. [ ] `pool.broadcast()` - Execute em todos os nodes
4. [ ] `RPyCExecutor` - Executor via RPyC over SSH

### P1 - Importante (Produção)
5. [ ] `_get_dlami()` - AMI via SSM
6. [ ] `_generate_user_data()` - Usar bootstrap DSL
7. [ ] `_check_instance_preemption()` - Implementar por provider
8. [ ] `_ping_instance()` - Health check via SSH
9. [ ] SSH key management no cluster state

### P2 - Nice to Have
10. [x] DigitalOcean provider ✅
11. [x] Vast.ai provider ✅
12. [x] Verda provider ✅
13. [ ] Instance type mapping dinâmico

### P3 - Futuro
13. [ ] Panel/visualization
14. [ ] Cost tracking
15. [ ] Torch/JAX/Keras integrations
16. [ ] Data utilities (samplers)

---

## Arquivos Modificados/Criados

```
skyward/v2/
├── __init__.py              ✅ Exports
├── events.py                ✅ Event definitions
├── bus.py                   ✅ AsyncEventBus
├── app.py                   ✅ @component, @on, @monitor
├── spec.py                  ✅ PoolSpec, ImageSpec
├── protocols.py             ✅ Transport, Executor protocols
├── node.py                  ✅ Node component
├── pool.py                  🟡 run/broadcast TBD
├── monitors.py              🟡 preemption/health stubs
├── bootstrap/
│   └── __init__.py          ✅ Re-export v1
├── transport/
│   ├── __init__.py          ✅ Exports
│   ├── ssh.py               ✅ SSHTransport
│   └── rpyc.py              ❌ TBD
└── providers/
    ├── __init__.py          ✅ Exports
    ├── aws/
    │   ├── __init__.py      ✅ Exports
    │   ├── config.py        ✅ AWS config
    │   ├── state.py         ✅ Cluster state
    │   ├── clients.py       ✅ Client[T] factories
    │   └── handler.py       🟡 bootstrap/AMI TBD
    ├── digitalocean/
    │   ├── __init__.py      ✅ Exports
    │   ├── config.py        ✅ DigitalOcean config
    │   ├── types.py         ✅ TypedDicts
    │   ├── state.py         ✅ Cluster state
    │   ├── client.py        ✅ pydo.aio async client
    │   └── handler.py       ✅ Event handlers
    ├── vastai/
    │   ├── __init__.py      ✅ Exports
    │   ├── config.py        ✅ VastAI config
    │   ├── types.py         ✅ TypedDicts
    │   ├── state.py         ✅ Cluster state
    │   ├── client.py        ✅ httpx async client
    │   └── handler.py       ✅ Event handlers
    └── verda/
        ├── __init__.py      ✅ Exports
        ├── config.py        ✅ Verda config
        ├── types.py         ✅ TypedDicts
        ├── state.py         ✅ Cluster state
        ├── client.py        ✅ httpx async + OAuth2
        └── handler.py       ✅ Event handlers
```

---

## Verificação

### Testes Unitários
```python
# Test event flow
async def test_cluster_lifecycle():
    async with app_context(AWSModule()) as app:
        pool = app.get(ComputePool)
        await pool.start()
        assert pool.is_ready
        await pool.stop()

# Test DI
def test_client_injection():
    injector = Injector([AWSModule()])
    ec2 = injector.get(Client[EC2Client])
    assert callable(ec2)  # É uma factory
```

### Teste Manual
```bash
uv run python -c "
import asyncio
from skyward.v2 import ComputePool, PoolSpec, ImageSpec, app_context, AWSModule

async def main():
    spec = PoolSpec(nodes=1, accelerator='T4', region='us-east-1')
    async with app_context(AWSModule()) as app:
        pool = app.get(ComputePool)
        async with pool:
            print(f'Cluster ready: {pool.cluster_id}')

asyncio.run(main())
"
```
