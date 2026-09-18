export type Kind = 'ready' | 'boot' | 'req' | 'gone' | 'off'

export const clamp = (v: number, a: number, b: number): number => Math.max(a, Math.min(b, v))
export const mean = (a: readonly number[]): number => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0)
export const median = (a: readonly number[]): number => {
  const s = a.slice().sort((x, y) => x - y)
  return s.length ? s[Math.floor(s.length / 2)]! : 0
}

/* ---------- what a node measures ---------- */

/**
 * One line of a chart: the readings it can be drawn from, best first, and what its numbers mean.
 *
 * A node reports whatever its image asked for — the collector's own names, or the ones a
 * ``skyward/worker/metrics.py`` builder gives — so a line names every reading that carries
 * the same quantity and draws the first one the node actually reports.
 */
export type Line = { names: readonly string[]; factor: number; label?: string }

/** One chart: a quantity, the scale it is drawn on, and the one or two lines on it. */
export type Gauge = {
  key: string
  group: 'Accelerator' | 'Host'
  label: string
  unit: string
  places: number
  /** the floor and the ceiling of the scale; a null ceiling is ``total``, or the highest value drawn */
  scale: readonly [number, number | null]
  /** the readings that say what the value is out of */
  total?: Line
  lines: readonly Line[]
}

const whole = (names: readonly string[]): Line => ({ names, factor: 1 })

/**
 * Every reading the node's collector takes, in the order the pages draw them.
 *
 * A percentage is drawn 0 to 100 and a memory against its own total, so the same chart at
 * two moments, or on two nodes, is the same picture. What a node reports that no gauge here
 * claims is a metric of its own — what ``sky.metrics.Custom`` declared — and is drawn under
 * its name, on a scale read off its values.
 */
export const GAUGES: readonly Gauge[] = [
  { key: 'accel', group: 'Accelerator', label: 'Utilization', unit: '%', places: 0, scale: [0, 100], lines: [whole(['gpu_util'])] },
  {
    key: 'vram',
    group: 'Accelerator',
    label: 'Memory',
    unit: 'GB',
    places: 0,
    scale: [0, null],
    total: { names: ['gpu_mem_total_mb'], factor: 1 / 1024 },
    lines: [{ names: ['gpu_mem_mb'], factor: 1 / 1024 }],
  },
  { key: 'temp', group: 'Accelerator', label: 'Temperature', unit: '°C', places: 0, scale: [30, 90], lines: [whole(['gpu_temp_c', 'gpu_temp'])] },
  { key: 'power', group: 'Accelerator', label: 'Power', unit: 'kW', places: 1, scale: [0, null], lines: [{ names: ['gpu_power_w'], factor: 1 / 1000 }] },
  { key: 'cpu', group: 'Host', label: 'CPU', unit: '%', places: 0, scale: [0, 100], lines: [whole(['cpu'])] },
  {
    key: 'ram',
    group: 'Host',
    label: 'Memory',
    unit: 'GB',
    places: 0,
    scale: [0, null],
    total: { names: ['mem_total_mb'], factor: 1 / 1024 },
    lines: [{ names: ['mem_used_mb'], factor: 1 / 1024 }],
  },
  { key: 'disk', group: 'Host', label: 'Disk', unit: '%', places: 0, scale: [0, 100], lines: [whole(['disk_used_pct'])] },
  {
    key: 'net',
    group: 'Host',
    label: 'Network',
    unit: 'MB/s',
    places: 0,
    scale: [0, null],
    lines: [
      { names: ['net_rx_kbps'], factor: 1 / 8000, label: 'in' },
      { names: ['net_tx_kbps'], factor: 1 / 8000, label: 'out' },
    ],
  },
]

/** The gauge a key names. */
export const gaugeOf = (key: string): Gauge | undefined => GAUGES.find((g) => g.key === key)

/** Every reading a gauge here claims, which is what makes everything else a custom metric. */
export const CLAIMED: ReadonlySet<string> = new Set(GAUGES.flatMap((g) => [...g.lines.flatMap((l) => l.names), ...(g.total?.names ?? [])]))

/** The value of one line off a set of readings: the first name it finds, in the unit it is drawn in. */
export const lineValue = (line: Line, readings: Readonly<Record<string, number>> | undefined): number | null => {
  const name = readings && line.names.find((n) => readings[n] !== undefined)
  return name === undefined ? null : readings![name]! * line.factor
}

/** What a gauge reads right now on one node, and what it is out of. */
export const gaugeValue = (g: Gauge, readings: Readonly<Record<string, number>> | undefined): number | null => lineValue(g.lines[0]!, readings)

/** A value in a gauge's unit, without the unit. */
export const figure = (g: Gauge, v: number): string => (g.places ? v.toFixed(g.places) : Math.round(v).toLocaleString('en-US'))

/** The gauge that says whether a machine is working: its accelerators where it has them, and its CPUs where it has none. */
export const loadGauge = (accelerated: boolean): Gauge => gaugeOf(accelerated ? 'accel' : 'cpu')!

/** How hard one node is working on that gauge, which is what tints its hexagon. */
export const loadOf = (readings: Readonly<Record<string, number>> | undefined, accelerated: boolean): number | null =>
  lineValue(loadGauge(accelerated).lines[0]!, readings)

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

/** Every task the compute was given, in any state — the daemon's count, not the page the store holds. */
export const callsOf = (c: Compute): number =>
  c.tasks.queued + c.tasks.running + c.tasks.succeeded + c.tasks.failed + c.tasks.cancelled + c.tasks.timed_out + c.tasks.indeterminate
/** The tasks that ended in an error: failed or timed out. */
export const failedOf = (c: Compute): number => c.tasks.failed + c.tasks.timed_out
/** The tasks that are over, however they turned out. */
export const finishedOf = (c: Compute): number => callsOf(c) - c.tasks.queued - c.tasks.running

export const CAUSE: Record<Ending['cause'], string> = { requested: 'someone asked for it', abandoned: 'nobody renewed its lease' }

/** What was bought, or asked for: the offer a compute is bound to, else the first spec it named. */
export type Bound = { accelerator: string | null; accelerator_count: number; kind: string; region: string | null; instance: string | null }

export const boundOf = (c: Compute): Bound | null => {
  const s = c.spec.specs[0]
  if (c.offer) return { accelerator: c.offer.accelerator, accelerator_count: c.offer.accelerator_count, kind: c.offer.kind, region: c.offer.region, instance: c.offer.instance_type }
  return s ? { accelerator: s.accelerator ?? null, accelerator_count: s.accelerator_count ?? 1, kind: s.provider.kind, region: s.region ?? null, instance: null } : null
}

/** Whether the machines have cards at all, which is what a page measures and prices them by. */
export const acceleratedOf = (c: Compute): boolean => !!boundOf(c)?.accelerator

/** ``8× H100``, or the instance type of a machine with no accelerator. */
export const machineOf = (c: Compute): string => {
  const b = boundOf(c)
  if (!b) return '—'
  if (!b.accelerator) return b.instance ?? 'CPU'
  return `${b.accelerator_count > 1 ? `${b.accelerator_count}× ` : ''}${b.accelerator.toUpperCase()}`
}

/** How many nodes a compute is, and the floor or the range it is held to. */
export const sizeOf = (c: Compute, nodes: readonly Node[], live: boolean): { nodes: number; note: string } => {
  const b = c.spec.nodes
  const floor = b.min ?? b.initial
  return {
    nodes: live ? b.initial : nodes.length || targetOf(c),
    note: b.max ? `elastic ${floor} to ${b.max}` : floor !== b.initial ? `floor ${floor}` : '',
  }
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

/**
 * How long a task has been running rather than waiting: from when it started to the finish, or to now.
 *
 * Each rank counts from its latest attempt to start, so the queue a retry went back into is not run
 * time; the ranks of a broadcast run together, so the span is from the first of them to the finish. A
 * task that has started nowhere has no run time yet — what it has is a wait, which is how long ago it
 * was submitted.
 */
export const runOf = (t: Task): number | null => {
  const starts = new Map<number, number>()
  for (const e of t.executions) if (ms(e.started_at) > (starts.get(e.rank) ?? 0)) starts.set(e.rank, ms(e.started_at))
  return starts.size ? Math.max(0, (t.finished_at ? ms(t.finished_at) : Date.now()) - Math.min(...starts.values())) : null
}

/** Where a task falls in the order a queue reads in: what is running, then what is waiting, then what is done. */
export const groupOf = (t: Task): number => (t.state === 'running' ? 0 : t.state === 'queued' ? 1 : 2)

/**
 * The order a list of tasks reads in, which is the daemon's ``order=state``.
 *
 * Running first and newest first, because the newest is the one still saying something; then
 * queued with the next to run at the top; then finished with the latest to finish first.
 */
export const byState = (a: Task, b: Task): number =>
  groupOf(a) - groupOf(b) ||
  (groupOf(a) === 1 ? ms(a.submitted_at) - ms(b.submitted_at) : groupOf(a) === 2 ? ms(b.finished_at) - ms(a.finished_at) : ms(b.submitted_at) - ms(a.submitted_at))

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
  const span = runOf(t) ?? (t.finished_at ? ms(t.finished_at) : Date.now()) - ms(t.submitted_at)
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

/** Where a flat-top cell sits, in units of the distance from a cell's centre to its corner. */
const centre = ([q, r]: Axial): readonly [number, number] => [1.5 * q, SQ3 * (r + q / 2)]

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
    const fromBase = (cell: Axial): number => {
      const [x, y] = centre(cell)
      return Math.abs(Math.atan2(y, x) - Math.PI / 2)
    }
    out.push(...cells.sort((a, b) => fromBase(a) - fromBase(b)).slice(0, left))
  }
  return out
}

const pts = (s: number, unit: readonly (readonly [number, number])[]): string => unit.map(([x, y]) => `${(s * x).toFixed(2)},${(s * y).toFixed(2)}`).join(' ')

const H = SQ3 / 2

/** The points of a flat-top hexagon of circumradius ``s``, centred on the origin. */
export const hexPts = (s: number): string => pts(s, [[1, 0], [0.5, H], [-0.5, H], [-1, 0], [-0.5, -H], [0.5, -H]])

/**
 * A node: the same hexagon with one chevron cut out of it, as the two pieces either side of the cut.
 *
 * The logo is this hexagon cut twice — ``>>``, the dispatch — and a node is what receives one, so it carries one ``>``.
 * The cut is a quarter of the circumradius wide and runs parallel to the two edges on the right.
 */
export const nodePts = (s: number): readonly [string, string] => [
  pts(s, [[-0.125, -H], [0.375, 0], [-0.125, H], [-0.5, H], [-1, 0], [-0.5, -H]]),
  pts(s, [[0.5, -H], [1, 0], [0.5, H], [0.125, H], [0.625, 0], [0.125, -H]]),
]

export type HiveLayout = { cells: readonly (readonly [number, number])[]; w: number; h: number }

/** Where each of ``n`` cells of circumradius ``s`` sits, and the box that holds them. */
export const hive = (n: number, s: number, gap = 0.12): HiveLayout => {
  const step = s * (1 + gap)
  const at = bloom(n).map((cell) => centre(cell).map((v) => v * step) as [number, number])
  const xs = at.map((p) => p[0])
  const ys = at.map((p) => p[1])
  const ox = Math.min(...xs) - s
  const oy = Math.min(...ys) - s * H
  return { cells: at.map(([x, y]) => [x - ox, y - oy] as const), w: Math.max(...xs) - ox + s, h: Math.max(...ys) - oy + s * H }
}

/** The cell radius that fits a hive of ``n`` in a box of ``room`` × ``tall``. */
export const hiveSize = (n: number, room = 980, tall = 640): number => {
  const k = ringsFor(n)
  return Math.min(room / ((3 * k + 2) * 1.12), tall / ((2 * k + 1) * SQ3 * 1.12))
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

/** What an hour of what this compute is bought by costs, and what one of those is: an accelerator, or a whole machine where there are none. */
export const perUnit = (c: Compute, nodes: readonly Node[]): { rate: number; unit: string } => {
  const priced = nodes.find((n) => (n.price_per_hour ?? 0) > 0)
  const accelerators = acceleratedOf(c) ? Math.max(1, boundOf(c)?.accelerator_count ?? 1) : 1
  return { rate: (priced?.price_per_hour ?? 0) / accelerators, unit: acceleratedOf(c) ? '/accelerator·h' : '/node·h' }
}
