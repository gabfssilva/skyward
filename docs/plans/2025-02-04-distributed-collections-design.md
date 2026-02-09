# Distributed Collections Design

Estruturas de dados distribuídas para Skyward usando Ray Actors.

## Motivação

Permitir comunicação entre tasks, cache distribuído e coordenação de estado global em workloads ML distribuídos.

## API

### Funções soltas

```python
import skyward as sky

@sky.pool(provider=sky.AWS(), nodes=4)
def main():
    cache = sky.dict("embeddings")
    progress = sky.counter("steps")
    results = sky.list("outputs")
    seen = sky.set("processed")
    tasks = sky.queue("work")
    sync = sky.barrier("epoch", n=4)
    lock = sky.lock("critical")
```

Cada função usa get-or-create: retorna estrutura existente ou cria nova.

### Acesso via pool explícito

```python
with sky.pool(...) as p:
    cache = p.dict("cache")
```

`sky.dict("name")` é atalho para `_get_active_pool().dict("name")`.

## Estruturas

### Dict

Key-value distribuído.

```python
d = sky.dict("cache")

# Sync (bloqueia)
d["key"] = value
v = d["key"]
del d["key"]
"key" in d
len(d)

# Async
await d.get_async("key")
await d.set_async("key", value)
await d.update_async({"a": 1, "b": 2})
await d.pop_async("key", default=None)
await d.clear_async()
await d.keys_async()
await d.values_async()
await d.items_async()
```

### List

Lista distribuída para acumular resultados.

```python
lst = sky.list("results")

# Sync
lst.append(value)
lst[0]
len(lst)

# Async
await lst.append_async(value)
await lst.extend_async([a, b, c])
await lst.pop_async()
await lst.slice_async(0, 10)
```

### Set

Conjunto distribuído para deduplicação.

```python
s = sky.set("seen")

# Sync
s.add(item)
item in s
len(s)

# Async
await s.add_async(item)
await s.discard_async(item)
await s.contains_async(item)
```

### Counter

Contador atômico distribuído.

```python
c = sky.counter("steps")

# Sync
c.increment()
c.increment(10)
c.decrement()
int(c)
c.value

# Async
await c.increment_async(n=1)
await c.decrement_async(n=1)
await c.reset_async(value=0)
await c.value_async()
```

### Queue

Fila FIFO distribuída.

```python
q = sky.queue("tasks")

# Sync
q.put(item)
item = q.get()          # bloqueia até ter item
item = q.get(timeout=5) # com timeout

# Async
await q.put_async(item)
await q.get_async(timeout=5.0)
q.empty()
len(q)
```

### Barrier

Barreira de sincronização.

```python
b = sky.barrier("sync", n=4)

# Sync
b.wait()    # bloqueia até n workers chegarem

# Async
await b.wait_async()

# Reutilizar
b.reset()
```

### Lock

Exclusão mútua distribuída.

```python
lock = sky.lock("critical")

# Context manager sync
with lock:
    # seção crítica

# Context manager async
async with lock:
    # seção crítica

# Manual
lock.acquire()
lock.release()
await lock.acquire_async()
await lock.release_async()
```

## Consistência

### Modos

| Modo | Write | Read |
|------|-------|------|
| `strong` | Espera confirmação do Actor | Vai ao Actor |
| `eventual` | Fire-and-forget | Vai ao Actor |

A diferença está apenas no write: `eventual` não espera confirmação, é mais rápido mas sem garantia de que completou.

### Defaults por estrutura

| Estrutura | Default | Permite eventual? |
|-----------|---------|-------------------|
| `dict` | `eventual` | sim |
| `list` | `eventual` | sim |
| `set` | `eventual` | sim |
| `counter` | `eventual` | sim |
| `queue` | `strong` | não |
| `barrier` | `strong` | não |
| `lock` | `strong` | não |

Queue, barrier e lock sempre usam strong por questões de semântica (FIFO, sincronização, exclusão mútua).

### Override

```python
cache = sky.dict("cache")                          # eventual (default)
important = sky.dict("checkpoints", consistency="strong")  # override
```

## Implementação

### Arquitetura

Cada estrutura é um Ray Actor independente.

```
┌─────────────────────────────────────────────────────────┐
│                      Ray Cluster                        │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ DictActor    │  │ CounterActor │  │ LockActor    │  │
│  │ "cache"      │  │ "steps"      │  │ "critical"   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│         ▲                 ▲                 ▲          │
│         │                 │                 │          │
│  ┌──────┴─────────────────┴─────────────────┴───────┐  │
│  │                   Worker nodes                    │  │
│  │  sky.dict("cache")  sky.counter("steps")  ...    │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Naming

Actors nomeados `skyward:{tipo}:{nome}`:

- `sky.dict("cache")` → Actor `skyward:dict:cache`
- `sky.counter("steps")` → Actor `skyward:counter:steps`

### Get-or-create

```python
def get_or_create_actor(actor_cls, name: str):
    full_name = f"skyward:{actor_cls.type}:{name}"
    try:
        return ray.get_actor(full_name)
    except ValueError:
        return actor_cls.options(name=full_name).remote()
```

### Proxy Pattern

Usuário recebe um proxy, não o Actor diretamente.

```python
class DictProxy:
    _actor: ray.ActorHandle
    _consistency: Literal["strong", "eventual"]

    def __getitem__(self, key):
        return ray.get(self._actor.get.remote(key))

    def __setitem__(self, key, value):
        ref = self._actor.set.remote(key, value)
        if self._consistency == "strong":
            ray.get(ref)
        # eventual: fire-and-forget

    async def get_async(self, key):
        return await self._actor.get.remote(key)

    async def set_async(self, key, value):
        ref = self._actor.set.remote(key, value)
        if self._consistency == "strong":
            await ref
```

### Lifecycle

Estruturas vivem enquanto o pool existir. No `__exit__` do pool, o registry destrói todos os Actors criados.

```python
class DistributedRegistry:
    _actors: dict[str, ray.ActorHandle]

    def cleanup(self):
        for name, actor in self._actors.items():
            ray.kill(actor)
        self._actors.clear()
```

### Serialização

Usa Ray default (cloudpickle). Ray faz otimizações automáticas para tensors e arrays grandes.

## Estrutura de arquivos

```
skyward/
├── distributed/
│   ├── __init__.py      # Re-exports: dict, list, set, counter, queue, barrier, lock
│   ├── actors.py        # DictActor, ListActor, SetActor, CounterActor, QueueActor, BarrierActor, LockActor
│   ├── proxies.py       # DictProxy, ListProxy, SetProxy, CounterProxy, QueueProxy, BarrierProxy, LockProxy
│   ├── registry.py      # DistributedRegistry, get_or_create
│   └── types.py         # Consistency = Literal["strong", "eventual"]
├── facade.py            # Integrar com SyncComputePool
└── __init__.py          # Re-export distributed functions
```

## Exemplos de uso

### Cache de embeddings

```python
import skyward as sky

@sky.compute
def process_batch(texts: list[str]):
    cache = sky.dict("embeddings")
    results = []

    for text in texts:
        if text in cache:
            emb = cache[text]
        else:
            emb = compute_embedding(text)
            cache[text] = emb
        results.append(emb)

    return results

@sky.pool(provider=sky.AWS(), nodes=4)
def main():
    batches = [batch1, batch2, batch3, batch4]
    results = [process_batch(b) for b in batches] @ sky
```

### Progresso distribuído

```python
@sky.compute
def train_epoch(data):
    progress = sky.counter("steps")

    for batch in data:
        loss = train_step(batch)
        progress.increment()

    return loss

@sky.compute
def monitor():
    progress = sky.counter("steps")
    total = sky.counter("total")

    while progress.value < total.value:
        print(f"Progress: {progress.value}/{total.value}")
        time.sleep(1)
```

### Barreira de sincronização

```python
@sky.compute
def distributed_training():
    info = sky.instance_info()
    sync = sky.barrier("epoch_done", n=info.total_nodes)

    for epoch in range(10):
        train_local()
        sync.wait()
        if info.is_head:
            aggregate_gradients()
        sync.wait()
```

### Fila de trabalho

```python
@sky.compute
def producer():
    q = sky.queue("tasks")
    for item in generate_items():
        q.put(item)

@sky.compute
def consumer():
    q = sky.queue("tasks")
    results = sky.list("results")

    while True:
        item = q.get(timeout=5)
        if item is None:
            break
        results.append(process(item))
```

### Lock para seção crítica

```python
@sky.compute
def safe_update():
    lock = sky.lock("checkpoint")
    cache = sky.dict("state")

    with lock:
        current = cache.get("value", 0)
        cache["value"] = current + 1
```
