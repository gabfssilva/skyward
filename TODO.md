# TODO — Paridade V1 → V2

O que a V1 (`main`) tem e a V2 ainda não tem. Itens marcados como *(parcial)* têm um substituto incompleto na V2.

## CLI

O comando `sky` inteiro sumiu (`skyward/cli/`, entry point `[project.scripts]`, dependência `cyclopts`).

- [x] Entry point `sky` no `pyproject.toml` — extra opt-in `cli=[cyclopts]` (uma lib de CLI não desce pro node); erro claro se faltar
- [x] `sky offers` — `list`, `fetch`, `summary` *(`query` não portado: na v1 era SQL cru contra um catálogo local; na v2 o catálogo vive atrás do daemon e não há endpoint SQL — os filtros estruturados do `list` cobrem o uso real)*
- [x] `sky providers` — `list`, `list --kinds`, `check` *(v2 guarda credenciais no daemon e nunca as devolve; `check` lê a linha do provider e expõe `last_error`/`offers_fetched_at` em vez de autenticar localmente como a v1)*
- [x] `sky config` — `path`, `show`, `validate` *(a v2 não tem subsistema TOML; os comandos reportam a resolução real de URL/database e checam `/v1/health/ready`)*
- [x] `sky server` — `start` (foreground/daemonizado + pidfile + log), `stop`, `status` — uvicorn adicionado ao extra `server`
- [x] `sky compute` — `create`, `list`, `get`, `delete`, `view`, `run`, `exec`, `ls`, `rm`, `upload`, `download`. Controller `files.py` + protocolo `ports.Files` ao lado do `Forwarder` — **não** pelo byte-proxy: fazer SFTP por ele seria SSH dentro de SSH e exigiria a chave do node no cliente. `run` é o sentido da v1 (script `.py` local vira task pelo caminho existente, closure picklado *by value*); `exec` é o shell. Seletor `all | <rank>`: a v2 não tem head. `download` recusa `all`
- [x] `sky log` / `export_log` — replay/`--follow` do SSE + export `.jsonl`/`.md` *(export `.ipynb` caído: exigia o mapa de fontes por task que a v2 não carrega no log de eventos)*
- [x] `sky monitor` (TUI sobre uma sessão viva) — attach-and-render: resolve a ref e entrega ao `watcher()`/dashboard que a v2 já tem (`sdk/console.py` + `sdk/live.py`); nenhuma TUI nova foi escrita
- [x] `sky notebook install|remove`
- [x] `sky new` / `sky sessions` / `sky status` / `sky stop` — na v2 o **compute é a sessão** (não há session store nem config), então são aliases finos sobre os endpoints de compute: `status [REF]` despacha list/get, `stop` delega o `delete` inteiro (revision + If-Match + Idempotency-Key), e `new` registra o *mesmo objeto função* do `compute create` sob um segundo nome — alias real, sem segunda assinatura pra sair de sincronia
- [x] `sky console` / `sky repl` (PTY interativo em um node) — PTY sobre os mesmos streams HTTP pareados do port-forwarding (`SshChannel.open_shell` → `Terminal` → `ShellController`); node **fixo** (não round-robin: um shell é uma pessoa em uma máquina), `repl` é `console` com o interpretador do `/opt/skyward/.venv`
- [x] `sky version`

## Notebook

- [x] Integração Jupyter (`skyward/notebook/`: kernelspec, `SkywardKernelProvisioner`, entry point `jupyter_client.kernel_provisioners`) — extra opt-in `notebook`. O kernel sobe na máquina como uma task *streaming* e os 5 canais ZMQ vêm pelo port-forward do daemon. **Limitações documentadas**: exige exatamente 1 node ready (o forward é round-robin, então kernel e portas cairiam em máquinas diferentes — precisaria de um forward fixado no node); a imagem precisa trazer `ipykernel` (o cliente não roda shell no node); sem auto-criação de pool a partir de config; o kernel ocupa um slot de worker enquanto vive.

## API pública / SDK

- [x] Target implícito `sky` — `pending >> sky` resolvendo o pool ativo via ContextVar
- [x] `gather(stream=True, ordered=...)` — resultados conforme completam *(v2 `gather` só aceita `*pendings`)*
- [x] `sky.Options` e os knobs operacionais — struct `Options` no `ComputeSpec` + dataclass SDK, threaded por `Connector→Runtimes→Node→SshChannel`; defaults reproduzem os valores hard-coded (aditivo). Client-only: `ready_timeout`/`shutdown_timeout`. `bootstrap_timeout` já vivia no `Image`; `retry_on_interruption` já é `ComputeSpec.retry`. Fora de escopo: `reconcile_tick_interval` (tick process-global, um loop pro daemon inteiro) e `cluster=False` (caminho standalone net-new, baixo valor)
- [x] `HealthChecker` — probe periódico remoto com substituição de node — `health_command`/`health_interval`/`health_failures` em `Options`; loop `Node._health` (irmão do `_watch`, iniciado quando o node fica utilizável) → N falhas consecutivas chamam `listener("lost")`, reusando o path de substituição existente (sem wiring novo no control-plane)
- [x] Auto-scale e scale to zero — scale-to-zero (`min=0`) e lazy start (`desired=0`) **corrigidos** (bugs de falsiness em `reconciler.bounds`/`_status` + wake `compute.changed` no submit); cooldown/idle-timeout expostos via `sky.Options`. O reconciler já era um autoscaler load-based (`_desired` dimensiona pela carga, clampado nos bounds) — não precisou de classe nova.
- [x] Campos de `Spec` por-spec: `disk_gb` ✅, `max_hourly_cost` ✅ e `architecture` ✅ — filtros de offer reais em `market._candidates`. Disk filtra a offer, custo filtra o *buy* (o teto é sobre o preço pago), e arquitetura é assimétrica de propósito: `None` no spec é curinga, `None` na offer **não** satisfaz pedido nenhum (uma offer que não diz sua arquitetura não pode provar que roda wheels arm64, e subir wheel x86 em máquina arm é node quebrado). Vocabulário em `protocol/architectures.py`, normalizado no `Offer.__post_init__` e não por adapter — é como `h100` e `h100-sxm` viraram aceleradores diferentes. Populado só onde a API já reporta: AWS (`ProcessorInfo.SupportedArchitectures`) e Scaleway (`spec.arch`); o resto fica `None`, que é a resposta honesta. Continuam **compute-level** por caberem no modelo de uma frota por compute: `allocation`/`ttl`/`plugins`, `ports` (via `Compute(ports=[...])`) e `volumes` (via FUSE).
- [x] `sky.containers` / `DockerImage` *(v2 tem só `Image.base: str`)* — `DockerImage(str)` com classmethods; flui direto pro `Image.base` sem coerção
- [x] `sky.accelerators` com auto-complete — hoje a resolução é dinâmica (`__getattr__` sobre o catálogo) e o usuário precisa adivinhar que `sky.accelerators.RTX_3090()` existe; gerar um `.pyi` a partir do catálogo com todas as factories tipadas

## Runtime (dentro da `@sky.function`)

- [x] `sky.redirect_output` + `CallbackWriter` (redirecionar stdout/stderr para callback)
- [x] `stdout(only="head")` e forma predicado `Callable[[Info], bool]` *(v2 só aceita rank int/tuple)*
- [x] Campos ricos de `InstanceInfo`: `workers_per_node`, `global_worker_index`, `total_workers`, `head_addr`/`head_port`, `job_id`, IPs dos peers (`peers`) *(accelerator-detail e `network` ficaram de fora: `Machine`/`Node` não carregam GPU/NIC no runtime — precisaria de plumbing de control-plane inexistente)*
- [x] `shard` multi-array (`shard(x, y)`), indexação numpy/torch, overrides `node`/`total_nodes` *(v2 aceita só uma `Sequence`)*

## Coleções distribuídas

- [x] `Consistency` (`"strong"`/`"eventual"`) e kwarg `consistency=` em `dict`/`set`/`counter`
- [x] `registry` / `DistributedRegistry` público

## Plugins

- [x] Plugin `jax`
- [x] Plugin `keras`
- [x] Plugin `cuml`
- [x] Plugin `sklearn`
- [x] Plugin `accelerate` (env + process-group em `run()`, não `accelerate launch` — incoerente com o worker persistente)
- [x] Plugin `mig` (particionamento NVIDIA MIG)
- [x] Plugin `mps` (CUDA MPS)
- [x] ~~API de plugin ad-hoc: `Plugin.create(name)` + fluent `with_*`~~ — **won't-port**: conflita com o modelo plugin-como-valor da v2 (msgpack, reconstruído no node por `kind`; sem callables/pickle no fio de controle). Subclasses de `Plugin` cobrem os casos.
- [x] Hook `bootstrap` (ops extras de shell) separado de `image`
- [x] ~~Hook `around_process` (lifecycle por subprocesso do executor)~~ — **won't-port**: redundante com o idioma `run()`+guard que todo built-in (torch/jax/cuml/accelerate/mig) usa; roda no processo que executa a task, exatamente a janela que `around_process` mirava.
- [x] ~~Transform de launch-command (`LaunchContext`/`LaunchCommand`)~~ — **won't-port**: era código morto na v1 (nenhum plugin built-in usou) e incoerente com o worker casty persistente da v2 (`accelerate launch`/`torchrun` forkam um script que termina; o worker vive pra receber tasks).

## Providers / Infra

- [x] Warm pools / snapshots de imagem prontos — `Image.content_hash()` (só o que o bootstrap instala: base/python/pip/apt/indexes + `source`) + protocolo estrutural `Bakeable` ao lado do `Preemptible`; bake em `Machines.bake` disparado pelo reconciler no primeiro node rank-0 `ready`, reuso em `Machines.bind`. **Opt-in** via `Image(warm=True)` — a v1 vazava toda imagem que commitava, e nada aqui desregistra o que cria; o que é criado leva tag `skyward:image=<hash>`. Recusa bake com `skyward="local"` (os bytes mudam a cada edição, o hash mentiria). Só Container (port direto do `docker commit`) e AWS: RunPod/Novita/VastAI bootam de *nome* de imagem Docker e não têm verbo pra snapshotar um pod. `Cluster.prebaked` **não** portado — `--allow-existing` + `_serving()` já cobrem, e sem quebrar quando o hash bate mas a máquina divergiu
- [x] Módulo de object storage do usuário (`skyward/storage/`: `Storage` com upload/download/ls/exists/rm + presets `R2`, `S3`, `GCS`, `Backblaze`, `Wasabi`, `Hyperstack`) — extra opt-in `storage=[aioboto3]`; preset Hyperstack de-acoplado (sem auto-provisão de chave)
- [x] Montagem FUSE de volumes S3 — `Volume` em `ComputeSpec` (a v2 tem uma frota por compute), protocolo estrutural `Mountable` → `Mount(binding_patch, phases)`, resolvido uma vez em `Machines.bind`: hints entram no binding *antes* do launch (RunPod) e as phases ficam em `Infrastructure.volumes` pro `Connector` ler. **Credencial nunca vai inline no fio** — resolvida no daemon a partir do `ProviderStore`, com blob digest (`storage_sha256`) como escape hatch pros buckets que ele não conhece (R2/Backblaze/Wasabi); é o que o `ErrorCode.secret_in_definition` existe pra recusar, e um teste faz grep do secret no spec servido. AWS não passa credencial nenhuma (geesefs `--iam` no instance profile); GCP e Hyperstack mintam chave efêmera no bind e revogam no `release`; RunPod é network-volume nativo (pod não tem `CAP_SYS_ADMIN`). geesefs **pinado** (a v1 puxava `/releases/latest`) e as phases encadeadas com `&&` (a v1 juntava com newline sem `set -e`, então um mount falho passava batido e o node subia com diretório vazio)
- [x] Port-forwarding para localhost com roteamento (`Port(remote, local, route)`, `TcpProxy`, política round-robin) — endpoint no daemon via streams HTTP pareados (uniforme embedded+remoto; sem dep nova; streaming de request body embedded resolvido)
- [x] `PreemptionChecker` como protocolo explícito — `Preemptible(Catalog)` interno + wiring em `Machines.resolve` (proativo, reusa o path `_lost`) + impl AWS via EC2; outros providers seguem reativos (detecção quando a máquina some)

## Observabilidade / Eventos / Offers

- [x] Pacote `skyward/observability/` — logger estilo loguru (`logger.bind(...)`) + `logging.py` (`LogConfig`, `LogLevel`, `setup_logging`, `teardown_logging`), stdlib-only. Enraizado em `skyward.log`, **não** em `skyward` (um logger não-propagante em `skyward` é ancestral dos loggers do daemon e engolia os registros deles). `metrics.py` não portado: `skyward/metrics.py` já é a superfície pública (`sky.metrics`) — duplicar daria dois caminhos de import pros mesmos builders
- [x] `sky.metrics.*` — `Default`/`CPU`/`GPU`/`Disk` como configuração pública *(v2 tem `MetricSpec` no schema, sem namespace/builders)*
- [x] `sky.time`

## Housekeeping (quebrado/desatualizado na V2)

- [x] Taskfile: targets mortos — repontados (`test:e2e:distributed`→`test_distributed_e2e.py`, `test:e2e:image`→`test:bootstrap`, jax/keras→`test:plugin:*`, `test:offers`→`test_offers_cache.py`) e removidos os sem equivalente (`test:e2e:{torch,joblib}`, `test:offers:integration`, `catalog`); adicionado `gen:accelerators`. Todo path verificado por stat.
- [x] Docs stale: reescritas ~31 páginas (`cli.md`, `notebook.md`, `reference/events.md`, `plugins/*`, guias) contra o código real — várias descreviam APIs **inexistentes** (o builder `Plugin.create`, hooks que a v2 não tem, params inventados)
- [x] Guides sem plugin correspondente: 07 (keras), 16 (cuml), 18 (mig) — os três plugins agora existem; guias reconciliados com a API real
- [x] `CLAUDE.md` descreve a árvore da v1 — reescrito para a v2 (estrutura, control plane via daemon HTTP, modelo de plugin como valor msgspec, protocolos de provider, runtime, superfície pública, CLI); guias de estilo preservados, só os paths corrigidos
