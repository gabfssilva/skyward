import type {
  Accelerator,
  Compute,
  FunctionRef,
  ComputeSpec,
  Ending,
  Execution,
  Image,
  Node,
  Offer,
  Options,
  Provider,
  ProviderKind,
  Spec,
  Task,
  WireError,
  Worker,
} from './client'

export const MOCK = import.meta.env.VITE_MOCK === '1'

/* ---------- a deterministic generator, the prototype's ---------- */

let seed = 7
const rnd = (): number => {
  seed = (seed * 1103515245 + 12345) % 2147483648
  return seed / 2147483648
}
const clamp = (v: number, a: number, b: number): number => Math.max(a, Math.min(b, v))
const jitter = (base: number, spread: number): number => clamp(base + (rnd() - 0.5) * spread, 0, 100)
const now = (): number => Date.now()
const iso = (ms: number): string => new Date(ms).toISOString()
const rid = (p: string): string => `${p}_${Math.floor(rnd() * 0xffffff).toString(16).padStart(6, '0')}`

const err = (message: string): WireError => ({ code: 'reconcile_failed', message, retryable: true })

const OPTIONS: Options = {
  autoscale_cooldown: 0,
  autoscale_idle_timeout: 120,
  cluster: null,
  default_compute_timeout: 0,
  health_command: null,
  health_failures: 3,
  health_function: null,
  health_initial_delay: 0,
  health_interval: 30,
  health_timeout: 15,
  provision_timeout: 300,
  ssh_connect_timeout: 240,
  ssh_reconnect_attempts: 30,
  ssh_retry_delay: 2,
  worker_timeout: 180,
}

const image = (base: string | null, pip: string[]): Image => ({
  apt: [],
  base,
  bootstrap_timeout: 900,
  env: {},
  excludes: [],
  includes: [],
  includes_sha256: null,
  metrics: null,
  pip,
  pip_indexes: [],
  python: '3.12',
  shell_vars: {},
  skyward: 'auto',
  warm: false,
})

const worker = (executor: Worker['executor'], concurrency: number): Worker => ({ buffer: 0, concurrency, executor, reuse: true })

const spec = (
  provider: string,
  accelerator: string,
  accelerator_count: number,
  cpus: number,
  memory_gb: number,
  region: string,
): Spec => ({
  accelerator,
  accelerator_count,
  architecture: null,
  cpus,
  disk_gb: null,
  max_hourly_cost: null,
  memory_gb,
  provider: { kind: provider, name: 'default' },
  region,
})

/* ---------- providers and offers ---------- */

const provider = (
  id: string,
  kind: string,
  name: string,
  offers_count: number,
  fetchedMinutes: number,
  ttl: number,
  config: Record<string, unknown>,
  error: string | null,
): Provider => ({
  config,
  created_at: iso(now() - 8.6e7),
  id,
  kind,
  last_error: error ? err(error) : null,
  name,
  offers_count,
  offers_fetched_at: iso(now() - fetchedMinutes * 60e3),
  offers_ttl_seconds: ttl,
})

const providers: Provider[] = [
  provider('pr_aa01', 'aws', 'default', 1284, 4, 3600, { region: 'us-east-1' }, null),
  provider('pr_aa02', 'vastai', 'default', 3910, 2, 900, {}, null),
  provider('pr_aa03', 'runpod', 'default', 212, 11, 1800, { cloud: 'SECURE' }, null),
  provider('pr_aa04', 'novita', 'default', 168, 7, 1800, {}, null),
  provider('pr_aa05', 'lambda', 'default', 46, 19, 3600, {}, null),
  provider('pr_aa06', 'gcp', 'research', 0, 96, 3600, { project_id: 'skyward-research' }, 'the service account key was rejected: invalid_grant'),
]

const KIND_FIELDS: Record<string, string[]> = {
  aws: ['access_key_id', 'secret_access_key'],
  gcp: ['service_account_json'],
  vastai: ['api_key'],
  runpod: ['api_key'],
  novita: ['api_key'],
  lambda: ['api_key'],
  hyperstack: ['api_key'],
  verda: ['api_key'],
  scaleway: ['access_key', 'secret_key'],
  vultr: ['api_key'],
  tensordock: ['api_key', 'api_token'],
  jarvislabs: ['api_key'],
  massedcompute: ['api_key'],
  salad: ['api_key'],
  container: [],
}

const providerKinds: ProviderKind[] = Object.entries(KIND_FIELDS).map(([kind, credential_fields]) => ({
  credential_fields,
  kind,
  offers_ttl_seconds: 1800,
}))

const VRAM: Record<string, number> = { b200: 192, h200: 141, h100: 80, a100: 80, l40s: 48, a10g: 24, l4: 24, mi300x: 192 }
const ARCH: Record<string, string> = {
  b200: 'Blackwell',
  h200: 'Hopper',
  h100: 'Hopper',
  a100: 'Ampere',
  l40s: 'Ada Lovelace',
  a10g: 'Ampere',
  l4: 'Ada Lovelace',
  mi300x: 'CDNA3',
}

const accelerators: Accelerator[] = Object.entries(VRAM).map(([name, vram]) => ({
  name,
  vram,
  manufacturer: name === 'mi300x' ? 'AMD' : 'NVIDIA',
  architecture: ARCH[name] ?? '',
  cuda_min: '',
  cuda_max: '',
}))

const O = (
  kind: string,
  instance_type: string,
  accelerator: string,
  accelerator_count: number,
  cpus: number,
  memory_gb: number,
  region: string,
  spot_price: number | null,
  on_demand_price: number,
  available: number,
  billing_unit: Offer['billing_unit'],
  fetchedMinutes: number,
): Offer => ({
  accelerator,
  accelerator_count,
  architecture: ARCH[accelerator] ?? null,
  available,
  billing_unit,
  cpus,
  disk_gb: null,
  expires_at: iso(now() + 1800e3),
  fetched_at: iso(now() - fetchedMinutes * 60e3),
  id: rid('of'),
  instance_type,
  kind,
  memory_gb,
  on_demand_price,
  price: spot_price ?? on_demand_price,
  provider_id: providers.find((p) => p.kind === kind)?.id ?? 'pr_unknown',
  provider_name: 'default',
  region,
  specific: {},
  spot_price,
  vram: VRAM[accelerator] ?? null,
})

const offers: Offer[] = [
  O('aws', 'p5.48xlarge', 'h100', 8, 192, 2048, 'us-east-1', 31.46, 98.32, 6, 'second', 4),
  O('aws', 'p4d.24xlarge', 'a100', 8, 96, 1152, 'us-east-1', 12.42, 32.77, 14, 'second', 4),
  O('aws', 'g6e.xlarge', 'l40s', 1, 4, 32, 'us-east-1', 0.74, 1.86, 40, 'second', 4),
  O('aws', 'g5.xlarge', 'a10g', 1, 4, 16, 'us-west-2', 0.42, 1.01, 62, 'second', 4),
  O('gcp', 'a3-highgpu-8g', 'h100', 8, 208, 1872, 'us-central1', 27.11, 88.24, 2, 'second', 96),
  O('gcp', 'a2-ultragpu-8g', 'a100', 8, 96, 1360, 'us-central1', 11.04, 31.78, 5, 'second', 96),
  O('vastai', 'host-42871', 'h100', 8, 128, 1024, 'eu-central', 11.68, 14.32, 3, 'minute', 2),
  O('vastai', 'host-51903', 'l4', 1, 8, 32, 'eu-central', 0.31, 0.44, 410, 'minute', 2),
  O('vastai', 'host-77120', 'l40s', 1, 16, 64, 'eu-central', 0.68, 0.84, 55, 'minute', 2),
  O('vastai', 'host-38442', 'mi300x', 8, 192, 1536, 'us-west', 9.84, 12.6, 1, 'minute', 2),
  O('runpod', 'H100 SXM5', 'h100', 8, 176, 1536, 'us-or-1', 15.84, 19.92, 8, 'minute', 11),
  O('runpod', 'B200', 'b200', 8, 224, 2880, 'us-tx-3', null, 51.12, 2, 'minute', 11),
  O('runpod', 'A10G', 'a10g', 1, 8, 32, 'us-or-1', null, 0.79, 90, 'minute', 11),
  O('runpod', 'L40S', 'l40s', 1, 16, 62, 'eu-ro-1', null, 1.14, 47, 'minute', 11),
  O('novita', 'H100-SXM-80GB', 'h100', 8, 128, 1024, 'eu-west', null, 18.32, 4, 'minute', 7),
  O('novita', 'L4-24GB', 'l4', 1, 8, 32, 'eu-west', null, 0.42, 220, 'minute', 7),
  O('lambda', 'gpu_8x_h100_sxm5', 'h100', 8, 208, 1800, 'us-east-3', null, 23.92, 0, 'minute', 19),
  O('lambda', 'gpu_1x_h200', 'h200', 1, 26, 225, 'us-south-1', null, 3.79, 9, 'minute', 19),
  O('hyperstack', 'n3-H100x8', 'h100', 8, 192, 1440, 'norway-1', null, 17.52, 5, 'minute', 26),
  O('verda', 'vd-h100x8', 'h100', 8, 160, 1024, 'eu-north', null, 16.8, 3, 'hour', 33),
  O('scaleway', 'H100-2-80G', 'h100', 2, 48, 480, 'fr-par-2', null, 5.42, 7, 'minute', 41),
]

/* ---------- the fleets ---------- */

type Quirk = { base?: number; state?: Node['state']; error?: string; address?: null }

type FleetOptions = {
  base: number
  net: string
  mach: string
  accelerator: string
  count: number
  price: number
  market: 'spot' | 'on_demand'
}

/** The live gauge behind one node, exactly the prototype's `n.m`. */
type Live = { gpu: number; vram: number; cpu: number; temp: number; net: number }

const live = new Map<string, Live>()

const mkNode = (computeId: string, rank: number, o: FleetOptions, q: Quirk): Node => {
  const state = q.state ?? 'ready'
  const address = q.address === null || state !== 'ready' ? null : `${o.net}.${2 + (rank % 250)}`
  const base = q.base ?? (rank % 17 === 3 ? 46 : o.base)
  const node: Node = {
    accelerator: o.accelerator,
    address,
    billing_unit: 'second',
    compute_id: computeId,
    created_at: iso(now() - 3.6e6),
    desired: 'present',
    generation: 1,
    id: `nd_${computeId.slice(4)}_${rank}`,
    last_error: q.error ? err(q.error) : null,
    launched_at: iso(now() - 3.6e6),
    machine: state === 'ready' || state === 'bootstrapping' ? `${o.mach}${rank.toString(36)}` : null,
    market: o.market,
    price_per_hour: o.price,
    provider_binding: {},
    rank,
    revision: 1,
    state,
    terminated_at: null,
  }
  if (state === 'ready') {
    live.set(`${computeId}/${rank}`, {
      gpu: jitter(base, 10),
      vram: jitter(base - 6, 8),
      cpu: jitter(34, 16),
      temp: 58 + rnd() * 22,
      net: rnd() * 90,
    })
  }
  return node
}

const fleet = (computeId: string, count: number, o: FleetOptions, quirks: Record<number, Quirk> = {}): Node[] =>
  Array.from({ length: count }, (_, rank) => mkNode(computeId, rank, o, quirks[rank] ?? {}))

const mkCompute = (
  id: string,
  name: string,
  state: Compute['status']['state'],
  generation: number,
  createdAgo: number,
  s: ComputeSpec,
  nodesTotal: number,
  nodesReady: number,
  leaseIn: number,
  ended: Ending | null = null,
): Compute => ({
  created_at: iso(now() - createdAgo),
  ended,
  generation,
  id,
  lease: { owner: 'gabs@studio', expires_at: iso(now() + leaseIn * 1000) },
  name,
  offer: offers.find((o) => o.kind === s.specs[0]?.provider.kind && o.accelerator === s.specs[0]?.accelerator) ?? null,
  revision: generation,
  spec: s,
  status: { last_error: null, nodes_ready: nodesReady, nodes_total: nodesTotal, observed_generation: generation, state },
})

const computeSpec = (partial: Partial<ComputeSpec> & Pick<ComputeSpec, 'nodes' | 'specs' | 'image' | 'worker'>): ComputeSpec => ({
  allocation: 'spot_if_available',
  delete_on_exit: true,
  desired: 'running',
  options: OPTIONS,
  plugins: [],
  retry: null,
  selection: 'cheapest',
  ttl: 600,
  volumes: [],
  ...partial,
})

const execution = (
  taskId: string,
  rank: number,
  ordinal: number,
  state: Execution['state'],
  ms: number,
  finished: boolean,
  message: string | null,
): Execution => ({
  error: message ? err(message) : null,
  finished_at: finished ? iso(now() - 1000) : null,
  id: `${taskId}_${ordinal}`,
  node_id: null,
  ordinal,
  rank,
  result_sha256: null,
  retry_of: null,
  started_at: iso(now() - 1000 - ms),
  state,
})

/** The prototype's function names, kept where the daemon keeps them: on the function. */
const functions: Record<string, FunctionRef> = {}

const sha = (name: string): string => {
  let h = 2166136261
  for (const ch of name) h = Math.imul(h ^ ch.charCodeAt(0), 16777619)
  const digest = (h >>> 0).toString(16).padStart(8, '0').repeat(8)
  functions[digest] = { codec: 'cloudpickle+lz4', created_at: iso(now() - 8.6e6), name, sha256: digest, size_bytes: 4096 + (h >>> 20) }
  return digest
}

const task = (
  id: string,
  computeId: string,
  fn: string,
  dispatch: Task['dispatch'],
  state: Task['state'],
  submittedAgo: number,
  finishedAgo: number | null,
  executions: Execution[],
): Task => ({
  args_sha256: 'a'.repeat(64),
  compute_id: computeId,
  correlation_id: null,
  deadline_at: null,
  dispatch,
  executions,
  finished_at: finishedAgo === null ? null : iso(now() - finishedAgo),
  function: sha(fn),
  generation: 1,
  id,
  result_sha256: null,
  retry: null,
  state,
  submitted_at: iso(now() - submittedAgo),
})

/* ---------- the four computes ---------- */

const C1 = 'cmp_7f31ab'
const C2 = 'cmp_9ac410'
const C3 = 'cmp_2b90de'
const C4 = 'cmp_51c7aa'
const C5 = 'cmp_18de44'
const C6 = 'cmp_e3b71f'

const nodes: Record<string, Node[]> = {
  [C1]: fleet(C1, 64, { base: 91, net: '10.0.4', mach: 'i-0a91f3c2b', accelerator: 'h100', count: 8, price: 31.46, market: 'spot' }, {
    37: { state: 'bootstrapping', address: null },
  }),
  [C2]: fleet(C2, 284, { base: 72, net: '176.9', mach: 'vh-', accelerator: 'l4', count: 1, price: 0.31, market: 'spot' }, {
    19: { state: 'failed', error: 'interrupted by the provider 6m into the run', address: null },
    140: { state: 'lost', error: 'the worker stopped answering; the machine is being replaced', address: null },
    201: { state: 'bootstrapping', address: null },
    202: { state: 'provisioning', address: null },
    83: { base: 22 },
    84: { base: 19 },
    85: { base: 24 },
  }),
  [C3]: fleet(C3, 8, { base: 68, net: '38.104.7', mach: 'rp-77c', accelerator: 'l40s', count: 1, price: 1.14, market: 'on_demand' }, {
    6: { state: 'failed', error: 'interrupted by the provider 40s after launch — the bid was outrun', address: null },
    7: { state: 'bootstrapping', address: null },
  }),
  [C4]: fleet(C4, 1, { base: 27, net: '213.173.98', mach: 'rp-3f8', accelerator: 'a10g', count: 1, price: 0.79, market: 'on_demand' }),
}

const gone = (computeId: string, count: number, o: FleetOptions, failed: Record<number, string> = {}): Node[] =>
  fleet(computeId, count, o).map((n) => ({
    ...n,
    state: failed[n.rank] ? ('failed' as const) : ('deleted' as const),
    address: null,
    last_error: failed[n.rank] ? err(failed[n.rank]!) : null,
  }))

nodes[C5] = gone(C5, 16, { base: 0, net: '10.0.9', mach: 'i-0cc41f7a2', accelerator: 'a100', count: 8, price: 12.24, market: 'spot' }, {
  3: 'interrupted by the provider 12m into the run',
})
nodes[C6] = gone(C6, 2, { base: 0, net: '38.104.9', mach: 'rp-91b', accelerator: 'a10g', count: 1, price: 0.79, market: 'on_demand' })

const ready = (id: string): number => nodes[id]!.filter((n) => n.state === 'ready').length

const computes: Compute[] = [
  mkCompute(
    C1,
    'llama-3-sft',
    'ready',
    2,
    13.4e6,
    computeSpec({
      allocation: 'spot_if_available',
      delete_on_exit: false,
      ttl: 600,
      nodes: { initial: 64, min: 56, max: null },
      specs: [spec('aws', 'h100', 8, 192, 2048, 'us-east-1')],
      image: image(null, ['transformers==4.57.1', 'trl', 'datasets', 'peft']),
      worker: worker('thread', 8),
      plugins: [
        { kind: 'torch', params: { backend: 'nccl' } },
        { kind: 'huggingface', params: {} },
      ],
    }),
    64,
    ready(C1),
    42,
  ),
  mkCompute(
    C2,
    'embed-fleet',
    'ready',
    1,
    4.1e6,
    computeSpec({
      allocation: 'spot',
      ttl: 900,
      nodes: { initial: 284, min: 240, max: 320 },
      specs: [spec('vastai', 'l4', 1, 8, 32, 'eu-central'), spec('novita', 'l4', 1, 8, 32, 'eu-west')],
      image: image(null, ['sentence-transformers', 'pyarrow']),
      worker: worker('loky', 4),
      plugins: [{ kind: 'huggingface', params: {} }],
    }),
    284,
    ready(C2),
    55,
  ),
  mkCompute(
    C3,
    'eval-sweep',
    'degraded',
    3,
    1.6e6,
    computeSpec({
      allocation: 'spot',
      ttl: 900,
      nodes: { initial: 8, min: 4, max: 8 },
      specs: [spec('runpod', 'l40s', 1, 16, 62, 'eu-ro-1')],
      image: image('nvidia/cuda:12.8.0-runtime-ubuntu24.04', ['vllm==0.11.2', 'lm-eval']),
      worker: worker('process', 2),
      plugins: [{ kind: 'torch', params: { backend: 'gloo' } }],
    }),
    8,
    ready(C3),
    31,
  ),
  mkCompute(
    C4,
    'scratch',
    'ready',
    1,
    9.4e5,
    computeSpec({
      allocation: 'on_demand',
      selection: 'first',
      ttl: 600,
      nodes: { initial: 1, min: 1, max: null },
      specs: [spec('runpod', 'a10g', 1, 8, 32, 'us-or-1')],
      image: image(null, ['polars']),
      worker: worker('thread', 4),
      plugins: [],
    }),
    1,
    ready(C4),
    19,
  ),
]

/** What the daemon still remembers of computes it has released. */
const retired: Compute[] = [
  mkCompute(
    C5,
    'sft-ablation-7b',
    'deleted',
    2,
    9.2e6,
    computeSpec({
      allocation: 'spot',
      ttl: 900,
      nodes: { initial: 16, min: 12, max: 16 },
      specs: [spec('aws', 'a100', 8, 96, 1024, 'us-west-2')],
      image: image(null, ['transformers==4.57.1', 'peft']),
      worker: worker('thread', 8),
      plugins: [{ kind: 'torch', params: { backend: 'nccl' } }],
    }),
    16,
    0,
    0,
    { at: iso(now() - (9.2e6 - 7.56e6)), calls: 1421, cause: 'requested', cost: 16 * 12.24 * 2.1, failed: 3 },
  ),
  mkCompute(
    C6,
    'nightly-eval',
    'deleted',
    1,
    2.7e6,
    computeSpec({
      allocation: 'on_demand',
      selection: 'first',
      ttl: 600,
      nodes: { initial: 2, min: 2, max: null },
      specs: [spec('runpod', 'a10g', 1, 8, 32, 'us-or-1')],
      image: image('skyward/base:3.12', ['lm-eval']),
      worker: worker('thread', 2),
      plugins: [],
    }),
    2,
    0,
    0,
    { at: iso(now() - (2.7e6 - 2.28e6)), calls: 12, cause: 'abandoned', cost: (2 * 0.79 * 2.28e6) / 3.6e6, failed: 0 },
  ),
]

const everything = (): Compute[] => [...computes, ...retired]

const tasks: Record<string, Task[]> = {
  [C1]: [
    task('tk_9d21c4', C1, 'train_step', 'all', 'running', 92e3, null, []),
    task('tk_9d1f80', C1, 'save_checkpoint', 'one', 'succeeded', 640e3, 601e3, []),
    task('tk_9d1a02', C1, 'train_step', 'all', 'succeeded', 1.24e6, 1.1e6, [
      execution('tk_9d1a02', 37, 1, 'failed', 41200, true, 'CUDA out of memory. Tried to allocate 2.44 GiB'),
      execution('tk_9d1a02', 37, 2, 'succeeded', 96400, true, null),
    ]),
  ],
  [C2]: [
    task('tk_c81d33', C2, 'embed_shard', 'stream', 'running', 2.4e6, null, []),
    task('tk_c81c02', C2, 'warm_model', 'all', 'succeeded', 3.9e6, 3.8e6, []),
  ],
  [C3]: [
    task('tk_a4f2e1', C3, 'evaluate', 'all', 'running', 310e3, null, []),
    task('tk_a4e9b7', C3, 'evaluate', 'one', 'failed', 900e3, 869e3, [
      execution('tk_a4e9b7', 6, 1, 'failed', 12800, true, 'the node was lost mid-execution'),
      execution('tk_a4e9b7', 6, 2, 'failed', 9400, true, 'the node was lost mid-execution'),
      execution('tk_a4e9b7', 0, 3, 'failed', 2100, true, "ValueError: dataset 'hellaswag' has no split 'test'"),
    ]),
  ],
  [C4]: [task('tk_e01a77', C4, 'peek', 'one', 'succeeded', 120e3, 118e3, [])],
  [C5]: [
    task('tk_71b3d0', C5, 'train_step', 'all', 'succeeded', 9.1e6, 3.2e6, []),
    task('tk_71b0ac', C5, 'save_checkpoint', 'one', 'failed', 3.4e6, 3.3e6, [
      execution('tk_71b0ac', 3, 1, 'failed', 8200, true, 'the node was lost mid-execution'),
    ]),
  ],
  [C6]: [task('tk_2c9e11', C6, 'evaluate', 'all', 'succeeded', 2.6e6, 2.5e6, [])],
}

/* ---------- the loggers, line for line ---------- */

type LogFn = (i: number) => { rank: number; text: string }

const LOGGERS: Record<string, LogFn> = {
  [C1]: (i) => ({
    rank: i % 7 === 0 ? 0 : Math.floor(rnd() * 64),
    text:
      rnd() < 0.04
        ? 'NCCL WARN Bootstrap: retrying connect to 10.0.4.38:29500'
        : `step ${1400 + i} | loss ${(1.31 - i * 0.0006 + rnd() * 0.02).toFixed(4)} | lr 2.8e-5 | ${(4.1 + rnd() * 0.4).toFixed(2)} s/it | tok/s ${(1.42e6 + rnd() * 4e4).toFixed(0)}`,
  }),
  [C2]: (i) => ({
    rank: Math.floor(rnd() * 284),
    text:
      rnd() < 0.03
        ? 'ERROR urllib3 retry 2/5 — the object store closed the connection'
        : `shard ${1800 + i}/4096 · ${(38 + rnd() * 9).toFixed(1)}k rows/s · queue ${Math.floor(rnd() * 12)}`,
  }),
  [C3]: (i) => ({
    rank: Math.floor(rnd() * 6),
    text:
      rnd() < 0.08
        ? "ValueError: dataset 'hellaswag' has no split 'test'"
        : `hellaswag ${(0.771 + rnd() * 0.02).toFixed(3)} · ${240 + i} of 2000 · ${(11 + rnd() * 3).toFixed(1)} it/s`,
  }),
  [C4]: (i) => ({ rank: 0, text: `read parquet chunk ${i} · 512k rows · ${(0.4 + rnd() * 0.2).toFixed(2)}s` }),
}

/* ---------- the fetch interceptor ---------- */

/** A page of a listing, as the daemon cuts one: ``total`` is what the filters matched, counted without the cursor. */
type Paged<T> = { items: T[]; next_cursor: string | null; total: number }

type OfferSort = 'price' | 'vram' | 'available'

type Order = (a: Offer, b: Offer) => number

/** The listings the daemon answers whole: no cursor, and nothing counted. */
const whole = <T,>(items: T[]): { items: T[]; next_cursor: null } => ({ items, next_cursor: null })

/**
 * One page of what the filters matched, newest first by ``moment``.
 *
 * ``cursor`` is the id of the item the previous page ended on — that is how the daemon
 * writes one — and the page picks up at what came before that item's moment.
 * ``next_cursor`` is set only when the page came out full.
 */
const paged = <T extends { id: string }>(matched: T[], moment: (item: T) => number, cursor: string | null, limit: number): Paged<T> => {
  const ordered = [...matched].sort((a, b) => moment(b) - moment(a))
  const pivot = cursor === null ? undefined : ordered.find((item) => item.id === cursor)
  const items = (pivot === undefined ? ordered : ordered.filter((item) => moment(item) < moment(pivot))).slice(0, limit)
  const last = items[items.length - 1]
  return { items, next_cursor: last && items.length === limit ? last.id : null, total: matched.length }
}

/** The states a compute still owes something in: what ``live=true`` lists, and ``live=false`` the rest of. */
const LIVE = new Set<Compute['status']['state']>(['requested', 'provisioning', 'ready', 'degraded', 'deleting'])

/** What one accelerator costs — the only comparison that holds between offers selling different numbers of them — with an offer that has no price at all ordered last. */
const unit = (o: Offer): number => (o.price ?? Infinity) / Math.max(o.accelerator_count, 1)

const spotted = (o: Offer): boolean => (o.spot_price ?? null) !== null

/** How each order runs, as the daemon runs it: cheapest per accelerator first, and the other two highest first. */
const ORDERS: Record<OfferSort, Order> = {
  available: (a, b) => (b.available ?? 0) - (a.available ?? 0),
  price: (a, b) => unit(a) - unit(b),
  vram: (a, b) => (b.vram ?? 0) - (a.vram ?? 0),
}

const order = (sort: string | null): Order => ORDERS[sort === 'vram' || sort === 'available' ? sort : 'price']

/** Every filter the catalog takes, over one offer: who sells it, what it holds, and what it costs. */
const matches = (o: Offer, query: URLSearchParams): boolean => {
  const provider = query.get('provider')
  const kind = query.get('kind')
  const accelerator = query.get('accelerator')
  const spot = query.get('spot')
  const minCount = Number(query.get('min_count') ?? 0)
  const minVram = Number(query.get('min_vram') ?? 0)
  const maxPrice = Number(query.get('max_price') ?? 0)
  return (
    (!provider || o.provider_id === provider || o.provider_name === provider) &&
    (!kind || o.kind === kind) &&
    (!accelerator || o.accelerator === accelerator.toLowerCase()) &&
    (!minCount || o.accelerator_count >= minCount) &&
    (!minVram || (o.vram ?? 0) >= minVram) &&
    (!maxPrice || (o.price ?? Infinity) <= maxPrice) &&
    (spot === null || spotted(o) === (spot === 'true'))
  )
}

const json = (value: unknown): Response =>
  new Response(JSON.stringify(value), { status: 200, headers: { 'content-type': 'application/json' } })

const notFound = (): Response => new Response(JSON.stringify({ code: 'not_found', message: 'no such resource', retryable: false }), { status: 404 })

function route(path: string, init: RequestInit | undefined): Response {
  const method = (init?.method ?? 'GET').toUpperCase()
  const [raw] = path.split('?')
  const parts = (raw ?? '').replace(/^\/v1\//, '').split('/')

  if (raw === '/v1/events') return stream(path.split('?')[1] ?? '', new Headers(init?.headers).get('last-event-id'), init?.signal)
  if (raw === '/v1/events/log') return logPage(path.split('?')[1] ?? '')
  if (raw === '/v1/health/live') return json({ live: true, version: '0.9.3' })
  if (raw === '/v1/provider-kinds') return json(providerKinds)
  if (raw === '/v1/accelerators') return json(accelerators)
  if (parts[0] === 'functions' && parts[1]) {
    const fn = functions[parts[1]]
    return fn ? json(fn) : notFound()
  }
  if (raw === '/v1/providers') {
    if (method === 'POST') return json(providers[0])
    return json(whole(providers))
  }
  if (parts[0] === 'providers' && parts[1]) {
    if (method === 'DELETE') return new Response(null, { status: 204 })
    const found = providers.find((p) => p.id === parts[1])
    return found ? json(found) : notFound()
  }
  if (parts[0] === 'offers') {
    const query = new URLSearchParams(path.split('?')[1] ?? '')
    const limit = query.get('limit')
    const matched = offers.filter((o) => matches(o, query)).sort(order(query.get('sort')))
    return json({ items: limit === null ? matched : matched.slice(0, Number(limit)), next_cursor: null, total: matched.length })
  }

  if (parts[0] === 'computes') {
    if (!parts[1]) {
      if (method === 'POST') return json(computes[0])
      const query = new URLSearchParams(path.split('?')[1] ?? '')
      const state = query.get('state')
      const live = query.get('live')
      const cause = query.get('cause')
      const matched = everything().filter(
        (c) =>
          (!state || c.status.state === state) &&
          (live === null || LIVE.has(c.status.state) === (live === 'true')) &&
          (!cause || c.ended?.cause === cause),
      )
      return json(paged(matched, (c) => Date.parse(c.created_at), query.get('cursor'), Number(query.get('limit') ?? 50)))
    }
    const c = everything().find((x) => x.id === parts[1])
    if (!c) return notFound()
    if (!parts[2]) {
      if (method === 'DELETE') return new Response(null, { status: 204 })
      return json(c)
    }
    if (parts[2] === 'nodes') {
      if (parts[3]) {
        if (method === 'DELETE') return new Response(null, { status: 204 })
        const n = (nodes[c.id] ?? []).find((x) => x.id === parts[3])
        return n ? json(n) : notFound()
      }
      return json(whole(nodes[c.id] ?? []))
    }
    if (parts[2] === 'generations') return json(whole([]))
    if (parts[2] === 'exec') return json({ exit_code: 0, stdout: '', stderr: '' })
    return notFound()
  }

  if (parts[0] === 'tasks') {
    if (!parts[1]) {
      const query = new URLSearchParams(path.split('?')[1] ?? '')
      const compute = query.get('compute')
      const state = query.get('state')
      const matched = (compute ? (tasks[compute] ?? []) : Object.values(tasks).flat()).filter((t) => !state || t.state === state)
      return json(paged(matched, (t) => Date.parse(t.submitted_at), query.get('cursor'), Number(query.get('limit') ?? 50)))
    }
    const t = Object.values(tasks)
      .flat()
      .find((x) => x.id === parts[1])
    if (!t) return notFound()
    if (parts[2] === 'executions') {
      if (method === 'POST') return json(t.executions[0] ?? execution(t.id, 0, 1, 'started', 0, false, null))
      return json(whole(t.executions))
    }
    if (method === 'DELETE') return new Response(null, { status: 204 })
    return json(t)
  }

  return notFound()
}

/* ---------- the event stream, as the daemon frames it ---------- */

type Feed = { compute: string | null; types: Set<string> | null; send: (chunk: string) => void }

/** What a frame carries of the fields the daemon stamps its row with, and filters the log by. */
type Payload = { compute?: string | null; at?: string; node?: string; content?: string; task?: string | null } & Record<string, unknown>

/** A frame the daemon keeps a row of, as its log serves it. */
type Recorded = { compute: string | null; entry: { sequence: number; type: string; at: string; data: Payload } }

/** What the daemon publishes without a row: it rides the stream under the last sequence, and never reaches the log. */
const PUBLISHED = new Set(['compute.cost', 'node.metrics', 'node.progress'])

const feeds = new Set<Feed>()
const record: Recorded[] = []
let sequence = 0

const wants = (feed: Feed, frame: string, compute: string | null): boolean => (!feed.types || feed.types.has(frame)) && (!feed.compute || feed.compute === compute)

const message = (id: number, frame: string, data: string): string => `id: ${id}\nevent: ${frame}\ndata: ${data}\n\n`

/** Tell every feed that wants it, and write down what the daemon would, stamped with the moment the payload carries, or now. */
const emit = (frame: string, payload: Payload): void => {
  const data = JSON.stringify(payload)
  const { compute = null, at } = payload
  if (!PUBLISHED.has(frame)) {
    sequence += 1
    record.push({ compute, entry: { sequence, type: frame, at: at ?? iso(now()), data: payload } })
  }
  for (const feed of feeds) if (wants(feed, frame, compute)) feed.send(message(sequence, frame, data))
}

/** An SSE body the fetch transport reads, filtered the way the daemon filters, resumed past ``Last-Event-ID`` when that names a sequence, and hung up when the fetch is aborted. */
function stream(search: string, lastEventId: string | null, signal: AbortSignal | null | undefined): Response {
  const query = new URLSearchParams(search)
  const types = query.getAll('types')
  const encoder = new TextEncoder()
  let feed: Feed | null = null
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      feed = {
        compute: query.get('compute'),
        types: types.length ? new Set(types) : null,
        send: (chunk) => controller.enqueue(encoder.encode(chunk)),
      }
      const after = Number(lastEventId ?? Infinity)
      for (const { compute, entry } of record) if (entry.sequence > after && wants(feed, entry.type, compute)) feed.send(message(entry.sequence, entry.type, JSON.stringify(entry.data)))
      feeds.add(feed)
      signal?.addEventListener('abort', () => {
        if (feed) feeds.delete(feed)
        controller.close()
      })
    },
    cancel() {
      if (feed) feeds.delete(feed)
    },
  })
  return new Response(body, { status: 200, headers: { 'content-type': 'text/event-stream' } })
}

/** Any one of the strings, in the line a node printed: an entry that printed nothing holds none of them. */
const said = (content: string | undefined, contains: string[]): boolean => content !== undefined && contains.some((text) => content.toLowerCase().includes(text))

/**
 * The recorded frames, newest first, a page at a time, filtered the way the daemon filters.
 *
 * ``cursor`` is the sequence the last page ended on. Every filter narrows the rows rather than
 * the page: ``node`` scopes it to one machine's output, and ``contains`` keeps the entries whose
 * printed line holds any one of the strings, case-insensitively.
 */
function logPage(search: string): Response {
  const query = new URLSearchParams(search)
  const compute = query.get('compute')
  const task = query.get('task')
  const node = query.get('node')
  const types = query.getAll('types')
  const contains = query.getAll('contains').map((text) => text.toLowerCase())
  const cursor = Number(query.get('cursor') ?? Infinity)
  const limit = Number(query.get('limit') ?? 200)
  const items = record
    .filter(
      (r) =>
        r.entry.sequence < cursor &&
        (!compute || r.compute === compute) &&
        (!task || r.entry.data.task === task) &&
        (!node || r.entry.data.node === node) &&
        (!types.length || types.includes(r.entry.type)) &&
        (!contains.length || said(r.entry.data.content, contains)),
    )
    .slice(-limit)
    .reverse()
    .map((r) => r.entry)
  const last = items[items.length - 1]
  return json({ items, next_cursor: last && items.length === limit ? String(last.sequence) : null })
}

const nodeId = (computeId: string, rank: number): string => `nd_${computeId.slice(4)}_${rank}`

const gauges = (computeId: string, rank: number, m: Live): void => {
  emit('node.metrics', { type: 'node.metrics', compute: computeId, node: nodeId(computeId, rank), name: 'gpu_util', value: Math.round(m.gpu) })
  emit('node.metrics', { type: 'node.metrics', compute: computeId, node: nodeId(computeId, rank), name: 'gpu_temp', value: Math.round(m.temp) })
  emit('node.metrics', { type: 'node.metrics', compute: computeId, node: nodeId(computeId, rank), name: 'cpu', value: Math.round(m.cpu) })
  emit('node.metrics', { type: 'node.metrics', compute: computeId, node: nodeId(computeId, rank), name: 'gpu_mem_total_mb', value: 81920 })
  emit('node.metrics', {
    type: 'node.metrics',
    compute: computeId,
    node: nodeId(computeId, rank),
    name: 'gpu_mem_mb',
    value: Math.round((m.vram / 100) * 81920),
  })
}

const cursors: Record<string, number> = {}
let beat = 0

function tick(): void {
  beat += 1
  for (const c of computes) {
    const list = (nodes[c.id] ?? []).filter((n) => n.state === 'ready')
    for (const n of list) {
      const key = `${c.id}/${n.rank}`
      const m = live.get(key)
      if (!m) continue
      m.gpu = clamp(m.gpu + (rnd() - 0.5) * 10, 2, 100)
      m.vram = clamp(m.vram + (rnd() - 0.5) * 3, 2, 100)
      m.cpu = clamp(m.cpu + (rnd() - 0.5) * 7, 1, 100)
      m.temp = clamp(m.temp + (rnd() - 0.5) * 2.5, 34, 92)
      m.net = clamp(m.net + (rnd() - 0.5) * 14, 0, 120)
    }
    const start = cursors[c.id] ?? 0
    const slice = list.slice(start, start + SLICE)
    cursors[c.id] = start + SLICE >= list.length ? 0 : start + SLICE
    for (const n of slice) {
      const m = live.get(`${c.id}/${n.rank}`)
      if (m) gauges(c.id, n.rank, m)
    }

    if (list.length && rnd() < 0.8) {
      const i = (lines[c.id] = (lines[c.id] ?? 0) + 1)
      const line = LOGGERS[c.id]?.(i)
      if (line) emit('node.console', { type: 'node.console', compute: c.id, node: nodeId(c.id, line.rank), content: line.text, task: null })
    }
  }

  if (beat % 3 === 1)
    for (const c of computes) {
      const rate = (nodes[c.id] ?? []).reduce((s, n) => s + (n.price_per_hour ?? 0), 0)
      const hours = (now() - Date.parse(c.created_at)) / 3.6e6
      emit('compute.cost', { type: 'compute.cost', compute: c.id, cost: rate * hours, nodes: ready(c.id), at: iso(now()) })
    }

  if (beat % 9 === 0) {
    const running = Object.values(tasks)
      .flat()
      .filter((t) => t.state === 'running')
    const t = running[Math.floor(rnd() * running.length)]
    if (t) emit('task.started', { type: 'task.state', compute: t.compute_id, task: t.id, state: 'started', attempt: 1 })
  }
}

const SLICE = 48
const lines: Record<string, number> = {}

/** Seed every ready node so the comb is coloured before the first tick lands. */
function seed_all(): void {
  for (const c of computes) {
    for (const n of nodes[c.id] ?? []) {
      const m = live.get(`${c.id}/${n.rank}`)
      if (m) gauges(c.id, n.rank, m)
    }
  }
}

/** The events the daemon would already have recorded before the page opened. */
function history(): void {
  const t = now()
  emit('compute.ready', { type: 'compute.ready', compute: C1, nodes_ready: 64, nodes_total: 64, generation: 2, at: iso(t - 13.1e6) })
  emit('task.failed', { type: 'task.state', compute: C1, task: 'tk_9d1a02', state: 'failed', attempt: 1, at: iso(t - 1.24e6) })
  emit('task.succeeded', { type: 'task.state', compute: C1, task: 'tk_9d1a02', state: 'succeeded', attempt: 2, at: iso(t - 1.1e6) })
  emit('node.bootstrapping', { type: 'node.state', compute: C1, node: nodeId(C1, 37), state: 'bootstrapping', error: null, at: iso(t - 2.2e5) })
  emit('compute.ready', { type: 'compute.ready', compute: C2, nodes_ready: 284, nodes_total: 284, generation: 1, at: iso(t - 3.9e6) })
  emit('node.failed', { type: 'node.state', compute: C2, node: nodeId(C2, 19), state: 'failed', error: 'interrupted by the provider', at: iso(t - 3.6e5) })
  emit('node.lost', { type: 'node.state', compute: C2, node: nodeId(C2, 140), state: 'lost', error: 'the worker stopped answering', at: iso(t - 1.8e5) })
  emit('compute.cost', { type: 'compute.cost', compute: C2, cost: 100.24, nodes: 282, at: iso(t - 9e4) })
  emit('node.failed', { type: 'node.state', compute: C3, node: nodeId(C3, 6), state: 'failed', error: 'the bid was outrun', at: iso(t - 4.4e5) })
  emit('compute.degraded', { type: 'compute.degraded', compute: C3, error: '6 of 8 nodes ready, floor is 4', at: iso(t - 4.3e5) })
  emit('task.started', { type: 'task.state', compute: C3, task: 'tk_a4f2e1', state: 'started', attempt: 1, at: iso(t - 3.1e5) })
  emit('task.started', { type: 'task.state', compute: C1, task: 'tk_9d21c4', state: 'started', attempt: 1, at: iso(t - 9.2e4) })
  emit('node.console', { type: 'node.console', compute: C5, node: nodeId(C5, 0), content: 'step 9600 | loss 0.9127 | lr 1.0e-5 | 3.88 s/it', task: 'tk_71b3d0', at: iso(t - 3.52e6) })
  emit('node.console', { type: 'node.console', compute: C5, node: nodeId(C5, 3), content: 'NCCL WARN Net : Connection closed by remote peer 10.0.9.5<46211>', task: 'tk_71b3d0', at: iso(t - 3.41e6) })
  emit('node.console', { type: 'node.console', compute: C5, node: nodeId(C5, 0), content: 'saving checkpoint-9600\nERROR rank 3 stopped answering, the save is retried', task: 'tk_71b0ac', at: iso(t - 3.39e6) })
  emit('compute.deleted', { type: 'compute.deleted', compute: C5, nodes_ready: 0, nodes_total: 16, at: iso(t - 1.64e6) })
}

/* ---------- installation ---------- */

/** Installs the fetch interceptor serving the prototype's data, event stream included. */
export function installMock(): void {
  if (!MOCK) return

  const original = globalThis.fetch.bind(globalThis)
  globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url
    const path = url.startsWith('http') ? new URL(url).pathname + new URL(url).search : url
    if (!path.startsWith('/v1/')) return original(input, init)
    const request = typeof input === 'object' && 'method' in input ? { method: input.method } : init
    return route(path, request)
  }

  setTimeout(history, 40)
  setTimeout(seed_all, 60)
  setInterval(tick, 1200)
}
