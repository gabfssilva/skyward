export type Kind = 'ready' | 'boot' | 'req' | 'gone' | 'off'

export type MetricKey = 'gpu' | 'vram' | 'cpu' | 'temp' | 'net'

export type NodeMetrics = Record<MetricKey, number>

export const METRICS: readonly (readonly [MetricKey, string])[] = [
  ['gpu', 'GPU'],
  ['vram', 'VRAM'],
  ['temp', 'Temp'],
  ['cpu', 'CPU'],
  ['net', 'Net'],
]

export const UNIT: Record<MetricKey, string> = { gpu: '%', vram: '%', cpu: '%', temp: '°C', net: ' MB/s' }
export const SCALE: Record<MetricKey, readonly [number, number]> = {
  gpu: [0, 100],
  vram: [0, 100],
  cpu: [0, 100],
  temp: [30, 92],
  net: [0, 120],
}

export const PHASES: readonly string[] = ['ssh', 'set env', 'apt install', 'setup uv', 'create venv', 'install deps', 'install skyward', 'start worker']

export const clamp = (v: number, a: number, b: number): number => Math.max(a, Math.min(b, v))
export const last = (a: readonly number[]): number => (a.length ? a[a.length - 1]! : 0)
export const mean = (a: readonly number[]): number => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0)
export const median = (a: readonly number[]): number => {
  const s = a.slice().sort((x, y) => x - y)
  return s.length ? s[Math.floor(s.length / 2)]! : 0
}

export const norm = (metric: MetricKey, v: number): number => {
  const [lo, hi] = SCALE[metric]
  return clamp(((v - lo) / (hi - lo)) * 100, 0, 100)
}

export const HEATS: readonly string[] = ['--h1', '--h2', '--h3', '--h4', '--h5', '--h6']
export const heat = (p: number): string => `var(${HEATS[p < 8 ? 0 : p < 28 ? 1 : p < 50 ? 2 : p < 70 ? 3 : p < 88 ? 4 : 5]})`

export const KIND = (s: string): Kind =>
  s === 'ready'
    ? 'ready'
    : s === 'requested' || s === 'queued'
      ? 'req'
      : ['provisioning', 'connecting', 'bootstrapping'].includes(s)
        ? 'boot'
        : ['draining', 'deleting', 'deleted'].includes(s)
          ? 'off'
          : 'gone'

export const KIND_FILL: Record<Kind, string> = {
  ready: 'var(--ok)',
  boot: 'var(--boot)',
  req: 'var(--warn)',
  gone: 'var(--bad)',
  off: 'var(--edge)',
}

export const hexCols = (n: number): number => clamp(Math.round(Math.sqrt(n * 2.6)), 2, 30)
export const hexSize = (n: number, cols?: number, room = 980): number => clamp(Math.floor(room / (cols || hexCols(n))) - 4, 15, 58)

export const tally = (states: readonly string[]): Partial<Record<Kind, number>> =>
  states.reduce<Partial<Record<Kind, number>>>((t, s) => {
    const k = KIND(s)
    t[k] = (t[k] ?? 0) + 1
    return t
  }, {})

const LEG: readonly (readonly [Kind, string, string])[] = [
  ['ready', 'ready', '--ok'],
  ['boot', 'bootstrap', '--boot'],
  ['req', 'requested', '--warn'],
  ['gone', 'failed / lost', '--bad'],
  ['off', 'draining', '--edge'],
]

/**
 * The legend, named after the states it is actually showing.
 *
 * One kind covers draining, deleting and deleted — the same grey — and a compute
 * the daemon has released is all of the third, which does not read as "draining".
 */
export const legendOf = (states: readonly string[]): readonly (readonly [Kind, string, string])[] => {
  const off = states.filter((s) => KIND(s) === 'off')
  return LEG.map(([kind, label, token]) => [kind, kind === 'off' && off.length && off.every((s) => s === 'deleted') ? 'deleted' : label, token] as const)
}

export const money = (n: number, d = 2): string => '$' + n.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d })
export const pct = (n: number): string => Math.round(n) + '%'

export const dur = (ms: number): string => {
  const s = Math.max(0, Math.floor(ms / 1000))
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  return h ? `${h}h ${String(m).padStart(2, '0')}m` : m ? `${m}m ${String(s % 60).padStart(2, '0')}s` : `${s}s`
}

export const ago = (ms: number): string => {
  const s = Math.floor((Date.now() - ms) / 1000)
  return s < 60 ? `${s}s ago` : s < 3600 ? `${Math.floor(s / 60)}m ago` : `${Math.floor(s / 3600)}h ago`
}

export const clock = (ms: number): string => new Date(ms).toLocaleTimeString('en-GB', { hour12: false })

/* ---------- wire-shape derivations ---------- */

import type { Compute, Execution, Node, Task } from '../api/client'

export type NodeState = Node['state']
export type ComputeState = Compute['status']['state']

export const ms = (iso: string | null | undefined): number => (iso ? Date.parse(iso) : 0)

export const nodeLive = (n: Node): boolean => !['failed', 'lost', 'deleted'].includes(n.state)
export const readyOf = (nodes: readonly Node[]): Node[] => nodes.filter((n) => n.state === 'ready')
export const rateOf = (nodes: readonly Node[]): number => nodes.filter(nodeLive).reduce((s, n) => s + (n.price_per_hour ?? 0), 0)
export const gpusOf = (nodes: readonly Node[]): number => readyOf(nodes).length
export const accrued = (c: Compute, nodes: readonly Node[]): number => (rateOf(nodes) * (Date.now() - ms(c.created_at))) / 3.6e6
export const targetOf = (c: Compute): number => c.spec.nodes.max ?? c.spec.nodes.initial

/** The shape of a compute: how many nodes it is, of what, where. Unloaded nodes fall back to the target. */
export const specLine = (c: Compute, nodes: readonly Node[]): string => {
  const s = c.spec.specs[0]
  const size = nodes.length || targetOf(c)
  if (!s) return `${size} nodes`
  const count = s.accelerator_count > 1 ? `${s.accelerator_count}× ` : ''
  const more = c.spec.specs.length > 1 ? ` +${c.spec.specs.length - 1}` : ''
  return `${size}× ${count}${(s.accelerator ?? '?').toUpperCase()} · ${s.provider.kind}${more} · ${c.spec.allocation.replace(/_/g, ' ')} · ${s.region ?? 'any'}`
}

export const hash01 = (s: string): number => {
  let h = 2166136261
  for (const ch of String(s)) h = Math.imul(h ^ ch.charCodeAt(0), 16777619)
  return ((h >>> 0) % 1000) / 1000
}

export const taskOf = (tasks: Record<string, Task[]>, id: string): { t: Task; computeId: string } | null => {
  for (const [computeId, list] of Object.entries(tasks)) {
    const t = list.find((x) => x.id === id)
    if (t) return { t, computeId }
  }
  return null
}

export type ExecRow = { rank: number; ordinal: number; state: Execution['state']; ms: number; error: string | null }

const execMs = (e: Execution): number => {
  const start = ms(e.started_at)
  if (!start) return 0
  return Math.max(0, (e.finished_at ? ms(e.finished_at) : Date.now()) - start)
}

/**
 * The per-rank executions of a task, filled in for the ranks the task covers.
 *
 * A task's recorded executions win; the remaining ranks of an ``all`` or ``stream``
 * dispatch get a synthesized row so the comb and the bars show the whole fleet.
 */
export const execsOf = (t: Task, nodes: readonly Node[]): ExecRow[] => {
  const real: ExecRow[] = t.executions.map((e) => ({
    rank: e.rank,
    ordinal: e.ordinal,
    state: e.state,
    ms: execMs(e),
    error: e.error?.message ?? null,
  }))
  if (t.dispatch === 'one') {
    if (real.length) return real
    const first = readyOf(nodes)[0]
    return [{ rank: first?.rank ?? 0, ordinal: 1, state: t.state === 'succeeded' ? 'succeeded' : 'started', ms: 0, error: null }]
  }
  const seen = new Set(real.map((e) => e.rank))
  const span = (t.finished_at ? ms(t.finished_at) : Date.now()) - ms(t.submitted_at)
  const filled = readyOf(nodes)
    .filter((n) => !seen.has(n.rank))
    .map<ExecRow>((n) => ({
      rank: n.rank,
      ordinal: 1,
      state: t.finished_at ? 'succeeded' : 'started',
      ms: Math.round(span * (t.finished_at ? 0.62 + hash01(t.id + n.rank) * 0.36 : 1)),
      error: null,
    }))
  return [...real, ...filled].sort((a, b) => a.rank - b.rank || a.ordinal - b.ordinal)
}
