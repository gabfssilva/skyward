import { create } from 'zustand'
import { api } from '../api/client'
import type { Compute, FunctionRef, Node, Offer, Provider, ProviderKind, Task } from '../api/client'
import { CONSOLE_FRAMES, HEAD, STATE_FRAMES, subscribe } from '../api/events'
import type { SkyEvent, Subscription } from '../api/events'
import { useEffect } from 'react'
import { clamp, median, ms, PHASES } from './model'
import type { MetricKey, NodeMetrics } from './model'
import type { NodeProgress } from './nodes'

export type LogLine = { seq: number; at: number; rank: number; level: 'info' | 'warn' | 'err'; text: string }

export type Sel = { computeId: string; rank: number } | null

export type DockTab = 'logs' | 'events' | 'tasks' | 'shell'

export type Sheet =
  | null
  | { kind: 'wizard' }
  | { kind: 'scale'; computeId: string }
  | { kind: 'ports'; computeId: string }
  | { kind: 'addProvider'; provider?: string }
  | { kind: 'confirm'; title: string; body: string; confirm: string; danger?: boolean; onConfirm: () => void }
  | { kind: 'palette' }

export type MarketFilters = { accel: string; market: string; sort: string }

/** One sample of the cluster's GPU spread, the shape the band chart draws. */
export type BandPoint = { min: number; med: number; max: number }

export type Ui = {
  sel: Sel
  metric: MetricKey
  dock: DockTab
  dockMin: boolean
  termNode: number
  logRank: 'all' | number
  logFollow: boolean
  market: MarketFilters
  sheet: Sheet
}

export type Entities = {
  computes: Compute[]
  /** the deleted ones, newest first — history, never part of the fleet */
  history: Compute[]
  nodes: Record<string, Node[]>
  tasks: Record<string, Task[]>
  providers: Provider[]
  providerKinds: ProviderKind[]
  offers: Offer[]
  events: SkyEvent[]
  logs: Record<string, LogLine[]>
  /** what a task's `function` sha256 stands for; `null` is a sha the daemon does not know */
  functions: Record<string, FunctionRef | null>
}

export type Store = Entities &
  Ui & {
    /** live metrics per node, keyed `${computeId}/${rank}` */
    metrics: Record<string, NodeMetrics>
    /** metric history per node+metric, keyed `${computeId}/${rank}/${metric}` */
    hm: Record<string, number[]>
    /** the raw gauge values the daemon reports, keyed `${computeId}/${rank}` */
    raw: Record<string, Record<string, number>>
    /** boot progress per node, keyed by node id */
    progress: Record<string, NodeProgress>
    /** cluster GPU spread per compute, 40 samples */
    hist: Record<string, BandPoint[]>
    /** fleet hourly rate over time, 48 samples */
    spend: number[]
    loading: boolean
    error: string | null
    setUi: (patch: Partial<Ui>) => void
    setEntities: (patch: Partial<Entities>) => void
    pick: (sel: Sel) => void
    openSheet: (sheet: Sheet) => void
    closeSheet: () => void
    /** buffer one event; the fold lands on the next frame, in one `set` */
    apply: (event: SkyEvent) => void
    load: () => Promise<void>
    reloadCompute: (computeId: string) => Promise<void>
    /** load what a deleted compute left behind, once */
    reloadHistory: (computeId: string) => Promise<void>
    reloadProviders: () => Promise<void>
    reloadOffers: () => Promise<void>
    live: () => Subscription
  }

const HISTORY = 32
const BAND = 40
const SPEND = 48
const LOGS_MAX = 2000
const EVENTS_MAX = 200

const EMPTY: NodeMetrics = { gpu: 0, vram: 0, cpu: 0, temp: 0, net: 0 }

const page = <T,>(p: { items: T[] }): T[] => p.items

export const useStore = create<Store>((set) => ({
  computes: [],
  history: [],
  nodes: {},
  tasks: {},
  providers: [],
  providerKinds: [],
  offers: [],
  events: [],
  logs: {},
  functions: {},

  sel: null,
  metric: 'gpu',
  dock: 'logs',
  dockMin: false,
  termNode: 0,
  logRank: 'all',
  logFollow: true,
  market: { accel: 'all', market: 'all', sort: 'price' },
  sheet: null,

  metrics: {},
  hm: {},
  raw: {},
  progress: {},
  hist: {},
  spend: [],
  loading: false,
  error: null,

  setUi: (patch) => set(patch),
  setEntities: (patch) => set(patch),
  pick: (sel) => set({ sel }),
  openSheet: (sheet) => set({ sheet }),
  closeSheet: () => set({ sheet: null }),

  apply: (event) => enqueue(event),

  load: async () => {
    set({ loading: true, error: null })
    try {
      const [computes, history, providers, providerKinds, offers] = await Promise.all([
        api.computes({ live: true }).then(page).then(alive),
        api.computes({ state: 'deleted' }).then(page).then(retired),
        api.providers().then(page),
        api.providerKinds(),
        api.offers().then(page),
      ])
      const nodes: Record<string, Node[]> = {}
      const tasks: Record<string, Task[]> = {}
      await Promise.all(
        computes.map(async (c) => {
          const [ns, ts] = await Promise.all([api.nodes(c.id).then(page), api.tasks({ compute: c.id }).then(page)])
          nodes[c.id] = ns.sort((a, b) => a.rank - b.rank)
          tasks[c.id] = ts
        }),
      )
      set({ computes, history, nodes, tasks, providers, providerKinds, offers, loading: false })
      void named(Object.values(tasks).flat())
    } catch (error) {
      set({ loading: false, error: error instanceof Error ? error.message : String(error) })
    }
  },

  reloadCompute: async (computeId) => {
    try {
      const compute = await api.compute(computeId)
      if (compute.status.state === 'deleted') return set((s) => retire(s, compute))
      const [ns, ts] = await Promise.all([api.nodes(computeId).then(page), api.tasks({ compute: computeId }).then(page)])
      set((s) => ({
        computes: s.computes.some((c) => c.id === computeId) ? s.computes.map((c) => (c.id === computeId ? compute : c)) : [...s.computes, compute],
        nodes: { ...s.nodes, [computeId]: ns.sort((a, b) => a.rank - b.rank) },
        tasks: { ...s.tasks, [computeId]: merged(s.tasks[computeId] ?? [], ts) },
      }))
      void named(ts)
    } catch {
      /* a compute that no longer answers is dropped by the next full load */
    }
  },

  reloadHistory: async (computeId) => {
    try {
      const [ns, ts] = await Promise.all([api.nodes(computeId, { include_terminal: true }).then(page), api.tasks({ compute: computeId }).then(page)])
      set((s) => ({ nodes: { ...s.nodes, [computeId]: ns.sort((a, b) => a.rank - b.rank) }, tasks: { ...s.tasks, [computeId]: merged(s.tasks[computeId] ?? [], ts) } }))
      void named(ts)
    } catch {
      /* what the daemon no longer keeps is simply not shown */
    }
  },

  reloadProviders: async () => {
    const [providers, providerKinds] = await Promise.all([api.providers().then(page), api.providerKinds()])
    set({ providers, providerKinds })
  },

  reloadOffers: async () => {
    set({ offers: await api.offers().then(page) })
  },

  live: () => subscribe(enqueue, { types: STATE_FRAMES, lastEventId: HEAD }),
}))

/* ---------- ingestion ---------- */

type Patch = Partial<Store>

const RELOAD_EVERY = 2000
const HIDDEN_EVERY = 100
const GAUGES_EVERY = 1000
const TASKS_MAX = 500

const queue: SkyEvent[] = []
let frame: number | null = null
let timer: ReturnType<typeof setTimeout> | null = null
const gauges: SkyEvent[] = []
let gaugeTimer: ReturnType<typeof setTimeout> | null = null
let seq = 0

/**
 * Take one event now, fold it later.
 *
 * A daemon watching a busy fleet says thousands of things a second, and a `set`
 * per fact is a render per fact. The queue drains once a frame — hidden tabs get
 * no frames, so they drain on a timer instead — and the whole batch costs one
 * `set` and one render.
 */
function enqueue(event: SkyEvent): void {
  if (event.data.type === 'node.metrics') {
    gauges.push(event)
    gaugeTimer ??= setTimeout(flushGauges, GAUGES_EVERY)
    return
  }
  queue.push(event)
  if (frame !== null || timer !== null) return
  if (typeof document !== 'undefined' && document.hidden) {
    timer = setTimeout(() => {
      timer = null
      flush()
    }, HIDDEN_EVERY)
    return
  }
  frame = requestAnimationFrame(() => {
    frame = null
    flush()
  })
}

/* A hidden tab is given no frames, so a frame asked for before it was hidden never comes. */
if (typeof document !== 'undefined') {
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden || frame === null) return
    cancelAnimationFrame(frame)
    frame = null
    timer = setTimeout(() => {
      timer = null
      flush()
    }, HIDDEN_EVERY)
  })
}

function flush(): void {
  const batch = queue.splice(0)
  if (!batch.length) return

  const state = useStore.getState()
  const live = new Set(state.computes.map((c) => c.id))
  const known = new Set([...live, ...state.history.map((c) => c.id)])
  const draft: Store = { ...state }
  const patch: Patch = {}
  const stale = new Set<string>()
  const printedLines = new Map<string, LogLine[]>()

  for (const event of batch) {
    const compute = event.compute
    if (compute && !known.has(compute)) {
      if (event.type.startsWith('compute.')) stale.add(compute)
      continue
    }
    const payload = event.data
    if (payload.type === 'node.console') {
      const lines = printed(draft, event, payload.compute, payload.node, payload.content)
      if (lines.length) printedLines.set(payload.compute, [...(printedLines.get(payload.compute) ?? []), ...lines])
      continue
    }
    const folded = fold(draft, event)
    if (folded) {
      Object.assign(draft, folded)
      Object.assign(patch, folded)
    }
    if (compute && live.has(compute) && (event.type === 'node.state' || event.type === 'task.state' || event.type.startsWith('compute.'))) stale.add(compute)
  }

  if (printedLines.size) {
    const logs = { ...draft.logs }
    for (const [cid, lines] of printedLines) logs[cid] = [...(logs[cid] ?? []), ...lines].slice(-LOGS_MAX)
    draft.logs = logs
    patch.logs = logs
  }

  if (Object.keys(patch).length) useStore.setState(patch)
  for (const id of stale) resync(id)
}

/** Fold the node gauges gathered since the last second, in the order they arrived, in one `set`. */
function flushGauges(): void {
  gaugeTimer = null
  const batch = gauges.splice(0)
  const draft: Store = { ...useStore.getState() }
  const patch: Patch = {}
  for (const event of batch) {
    if (event.data.type !== 'node.metrics') continue
    const folded = gauge(draft, event.data.compute, event.data.node, event.data.name, event.data.value, event.at)
    if (!folded) continue
    Object.assign(draft, folded)
    Object.assign(patch, folded)
  }
  if (Object.keys(patch).length) useStore.setState(patch)
}

/** One event, folded into the slices it touches. Nothing here reads or writes the store. */
function fold(s: Store, event: SkyEvent): Patch | null {
  const payload = event.data
  switch (payload.type) {
    case 'node.state':
      return {
        ...logged(s, event),
        nodes: {
          ...s.nodes,
          [payload.compute]: (s.nodes[payload.compute] ?? []).map((n) =>
            n.id === payload.node
              ? { ...n, state: payload.state, last_error: payload.error ? { code: 'reconcile_failed' as const, message: payload.error, retryable: false } : n.last_error }
              : n,
          ),
        },
      }
    case 'node.progress':
      return {
        ...logged(s, event),
        progress: {
          ...s.progress,
          [payload.node]: {
            phase: s.progress[payload.node]?.phase ?? null,
            completion: payload.completion ?? s.progress[payload.node]?.completion ?? null,
            phases_done: s.progress[payload.node]?.phases_done ?? 0,
          },
        },
      }
    case 'node.phase': {
      const index = PHASES.indexOf(payload.phase)
      const done = payload.event === 'completed' && index >= 0 ? index + 1 : (s.progress[payload.node]?.phases_done ?? 0)
      return {
        ...logged(s, event),
        progress: {
          ...s.progress,
          [payload.node]: { phase: payload.phase, completion: PHASES.length ? done / PHASES.length : null, phases_done: done },
        },
      }
    }
    default:
      return logged(s, event)
  }
}

/** The dock's event log, newest first, capped — never re-sorted. */
const logged = (s: Store, event: SkyEvent): Patch => ({ events: [event, ...s.events].slice(0, EVENTS_MAX) })

const printed = (s: Store, event: SkyEvent, computeId: string, nodeId: string, content: string): LogLine[] => {
  const rank = rankOf(s, computeId, nodeId) ?? 0
  return content
    .split('\n')
    .filter(Boolean)
    .map<LogLine>((text) => ({ seq: ++seq, at: event.at, rank, level: levelOf(text), text }))
}

const levelOf = (text: string): LogLine['level'] =>
  /\b(error|exception|traceback|fatal)\b/i.test(text) ? 'err' : /\bwarn/i.test(text) ? 'warn' : 'info'

const rankOf = (state: Store, computeId: string, nodeId: string): number | null => {
  const node = (state.nodes[computeId] ?? []).find((n) => n.id === nodeId)
  return node ? node.rank : null
}

const reloaded = new Map<string, number>()
const pending = new Map<string, ReturnType<typeof setTimeout>>()

/**
 * Ask the daemon what a compute looks like now, at most once every couple of seconds.
 *
 * A call inside the window is not dropped: one trailing reload is scheduled for when
 * the window ends, so the last change always reaches the store.
 */
function resync(computeId: string): void {
  if (pending.has(computeId)) return
  const wait = RELOAD_EVERY - (Date.now() - (reloaded.get(computeId) ?? 0))
  if (wait > 0) {
    pending.set(
      computeId,
      setTimeout(() => {
        pending.delete(computeId)
        resync(computeId)
      }, wait),
    )
    return
  }
  reloaded.set(computeId, Date.now())
  void useStore.getState().reloadCompute(computeId)
}

/**
 * A page of tasks laid over the ones already known.
 *
 * The daemon answers with its latest page, so a task that fell off it is kept as it
 * was last seen, and a task on it replaces the known one with the same id. The list
 * is newest first by submission and holds at most ``TASKS_MAX``.
 */
function merged(known: readonly Task[], latest: readonly Task[]): Task[] {
  const byId = new Map(known.map((t) => [t.id, t] as const))
  for (const t of latest) byId.set(t.id, t)
  return [...byId.values()].sort((a, b) => ms(b.submitted_at) - ms(a.submitted_at)).slice(0, TASKS_MAX)
}

/* ---------- function names ---------- */

const asking = new Set<string>()

/**
 * Learn what the shas these tasks name are called.
 *
 * A task carries the hash of its code, not a name — the name is metadata on the
 * function resource. Each distinct sha is asked for once, ever: the answer is
 * cached, and so is the daemon's not knowing it, which is what keeps a task list
 * of a thousand rows to a handful of requests and no loop.
 */
async function named(tasks: readonly Task[]): Promise<void> {
  const known = useStore.getState().functions
  const wanted = [...new Set(tasks.map((t) => t.function))].filter((sha) => sha && !(sha in known) && !asking.has(sha))
  if (!wanted.length) return
  for (const sha of wanted) asking.add(sha)
  const resolved = await Promise.all(
    wanted.map((sha) =>
      api
        .function(sha)
        .then((fn) => [sha, fn] as const)
        .catch(() => [sha, null] as const),
    ),
  )
  for (const sha of wanted) asking.delete(sha)
  useStore.setState((s) => ({ functions: { ...s.functions, ...Object.fromEntries(resolved) } }))
}

/* ---------- what the dock listens to ---------- */

const consoles = new Map<string, Subscription>()
const cursors = new Map<string, string>()

/**
 * Keep one console feed open per compute in view, and none for the rest.
 *
 * A feed remembers the last event it handed over, after it is closed, so reopening
 * resumes past the lines already folded; the first open still replays the history.
 *
 * Console output is the bulk of the log — a hundred thousand lines of it — so it is
 * never taken globally. What the dock is showing is what the client asks for.
 */
function watchConsoles(ids: readonly string[]): void {
  const wanted = new Set(ids)
  for (const [id, feed] of consoles) {
    if (wanted.has(id)) continue
    feed.close()
    consoles.delete(id)
  }
  for (const id of wanted) {
    if (consoles.has(id)) continue
    const onEvent = (event: SkyEvent): void => {
      const cursor = cursors.get(id)
      if (/^[1-9][0-9]*$/.test(event.id) && (cursor === undefined || Number(event.id) > Number(cursor))) cursors.set(id, event.id)
      enqueue(event)
    }
    consoles.set(id, subscribe(onEvent, { compute: id, types: CONSOLE_FRAMES, lastEventId: cursors.get(id) }))
  }
}

/** The dock's console feeds: the compute it is scoped to, or every live one. */
export function useConsoles(computeId: string | null): void {
  const computes = useStore((s) => s.computes)
  const ids = computeId ? [computeId] : computes.map((c) => c.id)
  const key = ids.join(',')
  useEffect(() => {
    watchConsoles(key ? key.split(',') : [])
  }, [key])
}

/* ---------- metric folding ---------- */

/**
 * Fold one raw gauge into the five the prototype draws.
 *
 * The daemon reports what ``skyward/worker/metrics.py`` names — ``gpu_util``,
 * ``gpu_mem_mb``, ``gpu_temp``, ``cpu``, ``net_rx_*``/``net_tx_*`` — and the UI
 * wants percentages and a rate, so VRAM is a ratio of two gauges and the network
 * is the delta of a cumulative counter.
 */
function gauge(s: Store, computeId: string, nodeId: string, name: string, value: number, at: number): Patch | null {
  const rank = rankOf(s, computeId, nodeId)
  if (rank === null) return null
  const key = `${computeId}/${rank}`
  const previous = s.raw[key] ?? {}
  const raw = { ...previous, [name]: value, [`${name}@`]: at }
  const sampled: Patch = { raw: { ...s.raw, [key]: raw } }

  const patch = derive(name, value, raw, previous, at)
  if (!patch) return sampled

  const next = { ...(s.metrics[key] ?? EMPTY), ...patch }
  const hm = { ...s.hm }
  for (const metric of Object.keys(patch) as MetricKey[]) {
    const series = `${key}/${metric}`
    hm[series] = [...(hm[series] ?? []), next[metric]].slice(-HISTORY)
  }
  const metrics = { ...s.metrics, [key]: next }
  const folded: Patch = { ...sampled, metrics, hm }
  return patch.gpu === undefined ? folded : { ...folded, hist: band({ ...s, metrics }, computeId) }
}

function derive(name: string, value: number, raw: Record<string, number>, previous: Record<string, number>, at: number): Partial<NodeMetrics> | null {
  if (name === 'cpu') return { cpu: clamp(value, 0, 100) }
  if (name === 'gpu_util') return { gpu: clamp(value, 0, 100) }
  if (name === 'gpu_temp') return { temp: value }
  if (name === 'gpu_mem_mb' || name === 'gpu_mem_total_mb') {
    const total = raw['gpu_mem_total_mb'] ?? 0
    return total > 0 ? { vram: clamp(((raw['gpu_mem_mb'] ?? 0) / total) * 100, 0, 100) } : null
  }
  if (name.startsWith('net_rx_') || name.startsWith('net_tx_')) {
    const before = previous[name]
    const then = previous[`${name}@`]
    if (before === undefined || then === undefined) return null
    const seconds = Math.max(0.001, (at - then) / 1000)
    return { net: clamp(Math.max(0, value - before) / seconds / 1e6, 0, 1e4) }
  }
  return null
}

function band(s: Store, computeId: string): Record<string, BandPoint[]> {
  const values = (s.nodes[computeId] ?? [])
    .filter((n) => n.state === 'ready')
    .map((n) => s.metrics[`${computeId}/${n.rank}`]?.gpu ?? 0)
  if (!values.length) return s.hist
  const point: BandPoint = { min: Math.min(...values), med: median(values), max: Math.max(...values) }
  return { ...s.hist, [computeId]: [...(s.hist[computeId] ?? seedBand(point)), point].slice(-BAND) }
}

/* ---------- what the store keeps ---------- */

/** A deleted compute is not part of the fleet: no card, no total, no metrics feed. */
const alive = (computes: readonly Compute[]): Compute[] => computes.filter((c) => c.status.state !== 'deleted')

/**
 * History, newest first.
 *
 * The wire has no deletion time — a ``Compute`` carries ``created_at`` and nothing
 * else about when it ended — so what orders the list is when it was created.
 */
const retired = (computes: readonly Compute[]): Compute[] => [...computes].sort((a, b) => ms(b.created_at) - ms(a.created_at))

/** Move a compute out of the fleet and into history, keeping what was loaded about it. */
const retire = (s: Store, compute: Compute): Patch => ({
  computes: s.computes.filter((c) => c.id !== compute.id),
  history: retired([compute, ...s.history.filter((c) => c.id !== compute.id)]),
})

/* ---------- selectors ---------- */

/** The metric history of one node, padded with its current value. */
export const historyOf = (state: Store, computeId: string, rank: number, metric: MetricKey): number[] => {
  const series = state.hm[`${computeId}/${rank}/${metric}`]
  if (series && series.length > 1) return series
  const value = state.metrics[`${computeId}/${rank}`]?.[metric] ?? 0
  return Array.from({ length: HISTORY }, () => value)
}

export const nodesOf = (state: Store, computeId: string): Node[] => state.nodes[computeId] ?? []

export const tasksOf = (state: Store, computeId: string): Task[] => state.tasks[computeId] ?? []

export const logsOf = (state: Store, computeId: string): LogLine[] => state.logs[computeId] ?? []

export const progressOf = (state: Store, nodeId: string): NodeProgress | undefined => state.progress[nodeId]

/** A band opens on a full window, drifting around the first reading, as the prototype's does. */
const seedBand = (p: BandPoint): BandPoint[] =>
  Array.from({ length: BAND }, (_, i) => {
    const drift = Math.sin(i / 6) * 5
    return { min: clamp(p.min + drift - 2, 0, 100), med: clamp(p.med + drift, 0, 100), max: clamp(p.max + drift, 0, 100) }
  })

export const bandOf = (state: Store, computeId: string): BandPoint[] => state.hist[computeId] ?? []

export const computeById = (state: Store, computeId: string): Compute | undefined =>
  state.computes.find((c) => c.id === computeId) ?? state.history.find((c) => c.id === computeId)

/** What a function is called, once the daemon has said; `null` until then, or if it never will. */
export const functionName = (state: Store, sha256: string): string | null => state.functions[sha256]?.name ?? null

/** What to print for a task's function: its name, or the head of its hash in the mono face. */
export const useFunctionLabel = (sha256: string): { text: string; mono: boolean } => {
  const name = useStore((s) => functionName(s, sha256))
  return name ? { text: name, mono: false } : { text: sha256.slice(0, 8), mono: true }
}

/** Whether a compute is still the daemon's to change, or only a record of one. */
export const isLive = (state: Store, computeId: string): boolean => state.computes.some((c) => c.id === computeId)

/** Every node's current value for one metric, across a compute's ready nodes. */
export const valuesOf = (state: Store, computeId: string, metric: MetricKey): number[] =>
  nodesOf(state, computeId)
    .filter((n) => n.state === 'ready')
    .map((n) => state.metrics[`${computeId}/${n.rank}`]?.[metric] ?? 0)

/** The fleet's hourly rate, recorded so the spend chart has a series. */
export const pushSpend = (rate: number): void =>
  useStore.setState((s) => ({
    spend: (s.spend.length ? [...s.spend, rate] : seedSpend(rate)).slice(-SPEND),
  }))

/** The chart opens on a full window, the way the prototype does, so it never draws a stub. */
const seedSpend = (rate: number): number[] =>
  Array.from({ length: SPEND }, (_, i) => rate * (1 + Math.sin(i / 7) * 0.04 - (SPEND - 1 - i) * 0.002))
