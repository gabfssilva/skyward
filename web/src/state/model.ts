export type Kind = 'ready' | 'boot' | 'req' | 'gone' | 'off'

export type MetricKey = 'gpu' | 'vram' | 'cpu' | 'temp' | 'rx' | 'tx'

export type NodeMetrics = Record<MetricKey, number>

export const METRICS: readonly (readonly [MetricKey, string])[] = [
  ['gpu', 'GPU'],
  ['vram', 'VRAM'],
  ['temp', 'Temp'],
  ['cpu', 'CPU'],
  ['rx', 'Net ↓'],
  ['tx', 'Net ↑'],
]

export const UNIT: Record<MetricKey, string> = { gpu: '%', vram: '%', cpu: '%', temp: '°C', rx: ' MB/s', tx: ' MB/s' }
export const SCALE: Record<MetricKey, readonly [number, number]> = {
  gpu: [0, 100],
  vram: [0, 100],
  cpu: [0, 100],
  temp: [30, 92],
  rx: [0, 120],
  tx: [0, 120],
}

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
  return s < 60 ? `${s}s ago` : s < 3600 ? `${Math.floor(s / 60)}m ago` : s < 172800 ? `${Math.floor(s / 3600)}h ago` : `${Math.floor(s / 86400)}d ago`
}

export const clock = (ms: number): string => new Date(ms).toLocaleTimeString('en-GB', { hour12: false })

/* ---------- wire-shape derivations ---------- */

import type { Compute, Ending, Execution, Node, Offer, Task } from '../api/client'

export type NodeState = Node['state']
export type ComputeState = Compute['status']['state']

export const ms = (iso: string | null | undefined): number => (iso ? Date.parse(iso) : 0)

export const nodeLive = (n: Node): boolean => !['failed', 'lost', 'deleted'].includes(n.state)

/**
 * Whether the daemon still has a link to this machine, which is all a terminal needs.
 *
 * Wider than ready: the link is dialled as soon as the machine has an address, and the
 * whole bootstrap happens behind it. Those are the minutes somebody most wants to be
 * inside a machine, so they are the minutes a shell is offered for.
 */
export const nodeHeld = (n: Node): boolean => ['connecting', 'bootstrapping', 'ready'].includes(n.state)

/**
 * One node per rank, the one holding it now.
 *
 * The daemon gives a replacement the rank of the node it replaced, and the list keeps
 * the dead one as history, so a rank is its live node when there is one and its newest
 * otherwise. Anything that names a node by its rank reads it from here.
 */
export const holdersOf = (nodes: readonly Node[]): Node[] => {
  const holds = (a: Node, b: Node): boolean => (nodeLive(a) === nodeLive(b) ? ms(a.created_at) > ms(b.created_at) : nodeLive(a))
  const byRank = new Map<number, Node>()
  for (const n of nodes) {
    const held = byRank.get(n.rank)
    if (!held || holds(n, held)) byRank.set(n.rank, n)
  }
  return [...byRank.values()].sort((a, b) => a.rank - b.rank)
}

export const holderOf = (nodes: readonly Node[], rank: number): Node | undefined => holdersOf(nodes).find((n) => n.rank === rank)
export const readyOf = (nodes: readonly Node[]): Node[] => nodes.filter((n) => n.state === 'ready')
export const rateOf = (nodes: readonly Node[]): number => nodes.filter(nodeLive).reduce((s, n) => s + (n.price_per_hour ?? 0), 0)
export const targetOf = (c: Compute): number => c.spec.nodes.max ?? c.spec.nodes.initial
export const endedAt = (c: Compute): number => ms(c.ended?.at ?? c.created_at)
export const ranOf = (c: Compute): number => endedAt(c) - ms(c.created_at)

export const CAUSE: Record<Ending['cause'], string> = { requested: 'someone asked for it', abandoned: 'nobody renewed its lease' }

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
 * Which machine an attempt is on: the node it went to, or the rank it was admitted under.
 *
 * ``rank`` says which node of a *broadcast* an attempt is, and a broadcast's ranks are
 * frozen at admission. A dispatch to one node names no rank unless the caller asked for
 * one, so for those the machine is the one the daemon recorded when it placed it.
 */
const rankOf = (e: Execution, nodes: readonly Node[]): number => nodes.find((n) => n.id === e.node_id)?.rank ?? e.rank

/**
 * The per-rank executions of a task, filled in for the ranks the task covers.
 *
 * A task's recorded executions win; the remaining ranks of an ``all`` or ``stream``
 * dispatch get a synthesized row so the comb and the bars show the whole fleet.
 */
export const execsOf = (t: Task, nodes: readonly Node[]): ExecRow[] => {
  const real: ExecRow[] = t.executions.map((e) => ({
    rank: rankOf(e, nodes),
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

/* ---------- hive geometry ---------- */

export const SQ3 = Math.sqrt(3)

/** How many rings a hive of ``n`` cells needs around its centre. */
export const ringsFor = (n: number): number => (n <= 1 ? 0 : Math.ceil((-3 + Math.sqrt(9 + 12 * (n - 1))) / 6))

const DIRS: readonly (readonly [number, number])[] = [
  [1, 0],
  [1, -1],
  [0, -1],
  [-1, 0],
  [-1, 1],
  [0, 1],
]

export type Axial = readonly [number, number]

const ring = (k: number): Axial[] => {
  const out: Axial[] = []
  let q = -k
  let r = k
  for (const [dq, dr] of DIRS)
    for (let j = 0; j < k; j++) {
      out.push([q, r])
      q += dq
      r += dr
    }
  return out
}

/** ``n`` axial cells, spiralling out from the centre ring by ring. */
export const spiral = (n: number): Axial[] => {
  const out: Axial[] = [[0, 0]]
  for (let k = 1; out.length < n; k++) out.push(...ring(k).slice(0, n - out.length))
  return out
}

/** Like ``spiral``, but a partial last ring fills from the bottom up, so the hive stands on its base. */
export const bloom = (n: number): Axial[] => {
  const out: Axial[] = [[0, 0]]
  for (let k = 1; out.length < n; k++) {
    const cells = ring(k)
    const left = n - out.length
    if (left >= cells.length) {
      out.push(...cells)
      continue
    }
    const fromBase = ([q, r]: Axial): number => Math.abs(Math.atan2(1.5 * r, SQ3 * (q + r / 2)) - Math.PI / 2)
    out.push(...cells.sort((a, b) => fromBase(a) - fromBase(b)).slice(0, left))
  }
  return out
}

/** The points of a pointy-top hexagon of circumradius ``s``, centred on the origin. */
export const hexPts = (s: number): string =>
  Array.from({ length: 6 }, (_, i) => {
    const a = (Math.PI / 180) * (60 * i - 90)
    return `${(s * Math.cos(a)).toFixed(2)},${(s * Math.sin(a)).toFixed(2)}`
  }).join(' ')

export type HiveLayout = { cells: readonly (readonly [number, number])[]; w: number; h: number }

/** Where each of ``n`` cells of circumradius ``s`` sits, and the box that holds them. */
export const hive = (n: number, s: number, gap = 0.12): HiveLayout => {
  const step = s * (1 + gap)
  const pts = bloom(n).map(([q, r]) => [SQ3 * step * (q + r / 2), 1.5 * step * r] as const)
  const xs = pts.map((p) => p[0])
  const ys = pts.map((p) => p[1])
  const minX = Math.min(...xs)
  const minY = Math.min(...ys)
  const ox = minX - (s * SQ3) / 2
  const oy = minY - s
  return { cells: pts.map(([x, y]) => [x - ox, y - oy] as const), w: Math.max(...xs) - ox + (s * SQ3) / 2, h: Math.max(...ys) - oy + s }
}

/** The cell radius that fits a hive of ``n`` in a box of ``room`` × ``tall``. */
export const hiveSize = (n: number, room = 980, tall = 640): number => {
  const k = ringsFor(n)
  return Math.min(room / ((2 * k + 1) * SQ3 * 1.12), tall / ((3 * k + 2) * 1.12))
}

/* ---------- dates ---------- */

export const HOUR = 3.6e6
export const DAY = 24 * HOUR

export const dateOf = (ms: number): string => new Date(ms).toLocaleDateString('en-GB', { day: 'numeric', month: 'short' }) + ' ' + clock(ms).slice(0, 5)

/* ---------- slots and work ---------- */

/** Worker slots per node: the executor's concurrency, one when unset. */
export const slotsOf = (c: Compute): number => Math.max(1, c.spec.worker?.concurrency ?? 1)

/** How many executions are running on one rank right now. */
export const busyOf = (tasks: readonly Task[], nodes: readonly Node[], rank: number): number =>
  tasks.filter((t) => t.state === 'running').reduce((s, t) => s + execsOf(t, nodes).filter((e) => e.rank === rank && e.state === 'started').length, 0)

/** What one accelerator-hour costs on an offer: the spot price when there is one, else on demand. */
export const offerPerGpu = (o: Offer): number => (o.spot_price ?? o.on_demand_price ?? o.price ?? 0) / Math.max(1, o.accelerator_count)

/** What one accelerator-hour costs on this compute: the first priced node, split by its cards. */
export const perGpu = (c: Compute, nodes: readonly Node[]): number => {
  const priced = nodes.find((n) => (n.price_per_hour ?? 0) > 0)
  return priced ? (priced.price_per_hour ?? 0) / Math.max(1, c.offer?.accelerator_count ?? c.spec.specs[0]?.accelerator_count ?? 1) : 0
}
