# Migração RPyC → Ray

Design para substituir RPyC por Ray como runtime de execução distribuída no Skyward.

## Decisões

| Aspecto | Decisão |
|---------|---------|
| Escopo | Ray como runtime, Skyward provisiona |
| Conexão | Ray Client via SSH tunnel |
| API | Mantida (`@compute`, `>>`, `@`) |
| Distributed training | Ray coordena, NCCL para GPU comms |
| Preemption | Skyward monitora, Ray reconecta |
| Topologia | Node 0 = head, outros = workers |
| Controle de nós | Custom resources (`node_0`, `node_1`, ...) |

## 1. Escopo da Mudança

**Responsabilidades que permanecem no Skyward:**
- Provisionamento de VMs (AWS, VastAI, etc.)
- Monitoramento de preemption
- Health checks
- Lifecycle management (start/stop/replace)
- Bootstrap (instalar dependências)
- Observability (logs, métricas, panel)

**Responsabilidades que vão para o Ray:**
- Execução remota de código
- Serialização de funções e dados
- Task scheduling
- Comunicação inter-nós

```
┌─────────────────────────────────────────────────────────────┐
│                         SKYWARD                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Provisioning│  │ Monitoring  │  │ Bootstrap           │  │
│  │ (providers) │  │ (preemption)│  │ (cloud-init + Ray)  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ SSH tunnel
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                           RAY                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Remote Exec │  │ Scheduling  │  │ Serialization       │  │
│  │ (ray.remote)│  │ (placement) │  │ (cloudpickle)       │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 2. API Pública

A API permanece 100% idêntica:

```python
import skyward as sky

@sky.compute
def train(data):
    import torch
    # treina modelo
    return model

@sky.pool(
    provider=sky.AWS(region="us-east-1"),
    accelerator="A100",
    nodes=4,
)
def main():
    result = train(data) >> sky          # executa em um nó
    results = train(data) @ sky          # broadcast para todos
    a, b = (task1() & task2()) >> sky    # paralelo
```

**Implementação interna muda:**

```python
# ANTES (RPyC)
class PendingCompute[T]:
    async def _execute_single(self, pool: ComputePool) -> T:
        node = pool.get_node()
        async with node.executor() as exec:  # RPyC via SSH tunnel
            return await exec.execute(self.fn, *self.args)

# DEPOIS (Ray)
class PendingCompute[T]:
    async def _execute_single(self, pool: ComputePool) -> T:
        node_id = pool.get_node_id()
        # Ray remote com placement no nó específico
        result = await ray.get(
            ray.remote(self.fn)
            .options(resources={f"node_{node_id}": 1})
            .remote(*self.args, **self.kwargs)
        )
        return result
```

## 3. Bootstrap

### Ops para Ray

```python
# skyward/bootstrap/ray.py

from skyward.bootstrap.ops import Op, run, pip

def ray_install(version: str = "2.9.0") -> Op:
    """Instala Ray com dependências."""
    return pip.install(f"ray[default]=={version}")

def ray_head_start(
    port: int = 6379,
    dashboard_port: int = 8265,
    client_port: int = 10001,
    num_gpus: int | None = None,
    resources: dict[str, float] | None = None,
) -> Op:
    """Inicia Ray head node."""
    cmd = [
        "ray", "start", "--head",
        f"--port={port}",
        f"--dashboard-port={dashboard_port}",
        f"--ray-client-server-port={client_port}",
    ]
    if num_gpus is not None:
        cmd.append(f"--num-gpus={num_gpus}")
    if resources:
        import json
        cmd.append(f"--resources='{json.dumps(resources)}'")
    return run(*cmd)

def ray_worker_start(
    head_address: str,
    num_gpus: int | None = None,
    resources: dict[str, float] | None = None,
) -> Op:
    """Inicia Ray worker node."""
    cmd = ["ray", "start", f"--address={head_address}"]
    if num_gpus is not None:
        cmd.append(f"--num-gpus={num_gpus}")
    if resources:
        import json
        cmd.append(f"--resources='{json.dumps(resources)}'")
    return run(*cmd)

def server_ops(
    node_id: int,
    head_ip: str | None,
    num_gpus: int,
    ray_version: str = "2.9.0",
) -> list[Op]:
    """
    Gera ops de bootstrap para Ray.

    - node_id=0: head node
    - node_id>0: worker node (precisa de head_ip)
    """
    ops = [ray_install(ray_version)]

    resources = {f"node_{node_id}": 1.0}

    if node_id == 0:
        ops.append(ray_head_start(
            num_gpus=num_gpus,
            resources=resources,
        ))
    else:
        if head_ip is None:
            raise ValueError("worker nodes require head_ip")
        ops.append(ray_worker_start(
            head_address=f"{head_ip}:6379",
            num_gpus=num_gpus,
            resources=resources,
        ))

    return ops
```

### Ordem de Bootstrap

```
Node 0 (head):
  1. Provisiona VM
  2. Bootstrap: apt, pip, ray_head_start
  3. NodeReady

Node 1..N (workers):
  1. Provisiona VM
  2. Aguarda Node 0 ter IP
  3. Bootstrap: apt, pip, ray_worker_start(head_ip)
  4. NodeReady
```

Workers dependem do head ter IP antes de iniciar Ray.

## 4. Executor

```python
# skyward/executor.py

from dataclasses import dataclass, field
from typing import Any, Callable
import asyncio

import ray

from skyward.transport.ssh import SSHTunnel

@dataclass
class Executor:
    """Executor Ray-based com conexão via SSH tunnel."""

    head_ip: str
    ssh_user: str
    ssh_key_path: str
    port: int = 10001

    _tunnel: SSHTunnel | None = field(default=None, init=False)
    _connected: bool = field(default=False, init=False)

    async def connect(self, timeout: float = 120.0) -> None:
        """Conecta ao cluster Ray via SSH tunnel."""
        if self._connected:
            return

        # Cria túnel SSH: localhost:random → head:10001
        self._tunnel = await SSHTunnel.create(
            host=self.head_ip,
            user=self.ssh_user,
            key_path=self.ssh_key_path,
            remote_port=self.port,
        )

        address = f"ray://localhost:{self._tunnel.local_port}"

        # ray.init é síncrono, roda em thread
        await asyncio.to_thread(ray.init, address)
        self._connected = True

    async def execute[T](
        self,
        fn: Callable[..., T],
        *args: Any,
        node_id: int | None = None,
        **kwargs: Any,
    ) -> T:
        """Executa função no cluster."""
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")

        remote_fn = ray.remote(fn)

        if node_id is not None:
            remote_fn = remote_fn.options(
                resources={f"node_{node_id}": 1}
            )

        ref = remote_fn.remote(*args, **kwargs)
        return await asyncio.to_thread(ray.get, ref)

    async def broadcast[T](
        self,
        fn: Callable[..., T],
        *args: Any,
        num_nodes: int,
        **kwargs: Any,
    ) -> list[T]:
        """Executa função em todos os nós."""
        refs = []
        for node_id in range(num_nodes):
            remote_fn = ray.remote(fn).options(
                resources={f"node_{node_id}": 1}
            )
            refs.append(remote_fn.remote(*args, **kwargs))

        return await asyncio.to_thread(ray.get, refs)

    async def disconnect(self) -> None:
        """Desconecta do cluster."""
        if self._connected:
            await asyncio.to_thread(ray.shutdown)
            self._connected = False

        if self._tunnel:
            await self._tunnel.close()
            self._tunnel = None
```

## 5. Distributed Training

Ray fica responsável apenas pela coordenação. Comunicação GPU-GPU usa NCCL direto:

```python
@sky.compute
def train_step(rank: int, world_size: int):
    import torch.distributed as dist

    # NCCL: GPU ↔ GPU direto (não passa pelo Ray)
    dist.all_reduce(gradients)

    return loss
```

```
┌─────────────────────────────────────────────────────────────┐
│                     Ray Cluster                             │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│  │ Node 0  │    │ Node 1  │    │ Node 2  │    │ Node 3  │  │
│  │ (head)  │    │         │    │         │    │         │  │
│  │ rank=0  │    │ rank=1  │    │ rank=2  │    │ rank=3  │  │
│  └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘  │
│       │              │              │              │        │
└───────┼──────────────┼──────────────┼──────────────┼────────┘
        │              │              │              │
        └──────────────┴──────────────┴──────────────┘
                         NCCL Ring
                    (GPU ↔ GPU direto)
```

### Integração com frameworks

```python
# skyward/integrations/torch.py

@sky.compute
def init_distributed(rank: int, world_size: int, master_addr: str):
    """Inicializa PyTorch distributed."""
    import os
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group(backend="nccl")

# Uso:
@sky.pool(nodes=4, accelerator="A100")
def main():
    head_ip = sky.instance_info().ip  # IP do head

    # Broadcast init para todos os nós
    sky.broadcast(init_distributed,
                  ranks=range(4),
                  world_size=4,
                  master_addr=head_ip)
```

## 6. Preemption e Recovery

Skyward continua monitorando preemption:

```
Preemption detectada (Skyward)
            │
            ▼
    InstancePreempted event
            │
            ▼
    Node.replace() → Nova VM
            │
            ▼
    Bootstrap (ray worker start)
            │
            ▼
    Worker reconecta ao cluster
            │
            ▼
    NodeReady event
```

Ray automaticamente reconecta workers. Tasks em execução no nó preemptado falham e podem ser retried.

## 7. Fluxo de Eventos

```
ClusterRequested
       │
       ▼
ClusterProvisioned
       │
       ▼
InstanceRequested (N vezes)
       │
       ▼
InstanceLaunched → InstanceRunning → InstanceProvisioned
       │
       ▼
BootstrapRequested
       │
       ▼
┌──────────────────────────────────────┐
│ Node 0: ray start --head             │
│ Node N: ray start --address=head:6379│
└──────────────────────────────────────┘
       │
       ▼
InstanceBootstrapped → NodeReady
       │
       ▼
ClusterReady
       │
       ▼
┌──────────────────────────────────────┐
│ SSH tunnel → ray.init("ray://...")   │
│ Conexão única, não por nó            │
└──────────────────────────────────────┘
```

## 8. Segurança

Conexão via SSH tunnel (mesmo padrão atual):

```
Controller ──── SSH tunnel ────► localhost:10001 ──► Head:10001
```

- Ray Client não exposto à internet
- Não precisa abrir portas extras no security group
- Reutiliza infraestrutura de túnel existente

## 9. Arquivos

### Criar

| Arquivo | Descrição |
|---------|-----------|
| `skyward/executor.py` | Novo Executor (Ray-based) |
| `skyward/bootstrap/ray.py` | Ops de bootstrap Ray |

### Modificar

| Arquivo | Mudança |
|---------|---------|
| `skyward/pool.py` | Usar novo Executor |
| `skyward/bootstrap/unified.py` | Integrar ray ops |
| `skyward/bootstrap/worker.py` | Remover rpyc, usar ray |
| `pyproject.toml` | ray[default], remover rpyc |

### Remover

| Arquivo | Motivo |
|---------|--------|
| `skyward/rpc/` | Diretório inteiro (RPyC) |

## 10. Dependências

```toml
# pyproject.toml

[project]
dependencies = [
    # Adicionar
    "ray[default]>=2.9.0",

    # Remover
    # "rpyc>=6.0.0",
]
```

## 11. Ordem de Implementação

1. `bootstrap/ray.py` - Ops de head/worker start
2. `executor.py` - Novo Executor com Ray Client
3. `bootstrap/unified.py` - Integrar ray ops
4. `pool.py` - Usar novo Executor
5. Remover `rpc/` - Limpar código RPyC
6. Testes - Validar cluster funciona
