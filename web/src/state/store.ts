import { create } from 'zustand'
import { api } from '../api/client'
import type { Compute, Ending, FunctionRef, LogEntry, LogQuery, Node, Offer, OfferSort, Page, Provider, ProviderKind, Task } from '../api/client'
import { CONSOLE_FRAMES, HEAD, STATE_FRAMES, recorded, subscribe } from '../api/events'
import type { SkyEvent, Subscription } from '../api/events'
import { useEffect, useMemo } from 'react'
import { clamp, endedAt, median, ms, PHASES } from './model'
import type { MetricKey, NodeMetrics } from './model'
import type { NodeProgress } from './nodes'

/** One printed line: ``sequence`` and ``part`` place it in the daemon's order, and ``rank`` is ``null`` until the store has the row of the node that printed it. */
export type LogLine = { compute: string; sequence: number; part: number; node: string; rank: number | null; at: number; level: 'info' | 'warn' | 'err'; text: string }

/** What a log view is looking at: one compute or every one, one node or all of them, and what a search is for. */
export type LogScope = { compute?: string; node?: string; contains?: readonly string[] }

/**
 * One scope's console: the newest page of the log, the older pages somebody asked for, and the stream since.
 *
 * ``cursor`` is where the next older page starts, and null once the log has nothing older
 * to give. ``held`` counts the lines read off the log rather than the stream — what the
 * live cap must not trim away under a reader who went looking for them.
 */
export type LogFeed = { key: string; scope: LogScope; lines: LogLine[]; cursor: string | null; loading: boolean; held: number }

/** A scope's recorded events, newest first, as far back as somebody asked. */
export type EventWindow = { items: SkyEvent[]; cursor: string | null; loading: boolean }

/** The tasks the daemon answered with under one set of filters, and how many matched them. */
export type TaskFeed = { key: string; items: Task[]; cursor: string | null; loading: boolean; total: number | null }

/** Where a paged listing stands: what continues it, whether a page is on its way, and how many rows the filters match. */
export type Pages = { key: string; cursor: string | null; loading: boolean; total: number | null }

/** How much of the ordered catalog the market holds, and how much of it there is. */
export type Catalog = { key: string; limit: number; loading: boolean; total: number | null }

export type Sel = { computeId: string; rank: number } | null

export type Sheet =
  | null
  | { kind: 'wizard' }
  | { kind: 'scale'; computeId: string }
  | { kind: 'ports'; computeId: string }
  | { kind: 'addProvider'; provider?: string }
  | { kind: 'confirm'; title: string; body: string; confirm: string; danger?: boolean; onConfirm: () => void }
  | { kind: 'palette' }

export type MarketFilters = { accel: string; market: 'all' | 'spot'; sort: OfferSort }

/** What the Tasks view is filtered by — both of them the daemon's to apply, so paging walks the filter. */
export type TaskFilters = { compute: 'all' | string; state: 'all' | Task['state'] }

/** What the Activity view is looking at. */
export type ActivityFilters = {
  kind: 'logs' | 'events'
  compute: 'all' | string
  rank: 'all' | number
  level: string
  q: string
  since: '5m' | '15m' | '1h' | 'all'
}

/** What the History card on the home page is filtered by. */
export type HistoryFilters = {
  q: string
  cause: 'all' | Ending['cause']
  provider: string
  accel: string
  since: '24h' | '7d' | '30d' | 'all'
}

/** One sample of the cluster's GPU spread, the shape the band chart draws. */
export type BandPoint = { min: number; med: number; max: number }

export type Ui = {
  sel: Sel
  metric: MetricKey
  /** the node page's shell card is open */
  shell: boolean
  logFollow: boolean
  act: ActivityFilters
  hist: HistoryFilters
  task: TaskFilters
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
  /** the events read off the log for a scope, newest first — keyed by compute and kind, `''` for every compute */
  windows: Record<string, EventWindow>
  /** the console one view is following, and how far back it has been read */
  feed: LogFeed | null
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
    bands: Record<string, BandPoint[]>
    /** what each live compute has cost so far, as the daemon's meter last said */
    costs: Record<string, number>
    loading: boolean
    error: string | null
    /** the tasks under the filters the Tasks view asked for: `null` before its first page */
    taskFeed: TaskFeed | null
    /** where the history listing stands, under the cause it was asked for */
    histPages: Pages | null
    /** how much of the ordered catalog the market has asked for */
    offerPages: Catalog | null
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
    /** learn the computes these ids name, for the ones no page has brought in */
    learn: (ids: readonly string[]) => Promise<void>
    /** read the page of console before the oldest line held */
    pageLogs: () => Promise<void>
    /** read the page of events before the oldest one held for this scope */
    pageEvents: (computeId: string | null, kind: string) => Promise<void>
    /** read the next page of tasks under the current filters, or the first page of new ones */
    pageTasks: (reset?: boolean) => Promise<void>
    /** read the next page of what has ended, or the first page under a new cause */
    pageHistory: (reset?: boolean) => Promise<void>
    /** ask the catalog for more of the order it is already in */
    moreOffers: () => Promise<void>
    reloadProviders: () => Promise<void>
    reloadOffers: () => Promise<void>
    live: () => Subscription
  }

const HISTORY = 32
const BAND = 40
const LOGS_MAX = 2000
const EVENTS_MAX = 200
const HISTORY_PAGE = 50
const OFFERS_PAGE = 200

const EMPTY: NodeMetrics = { gpu: 0, vram: 0, cpu: 0, temp: 0, net: 0 }

const page = <T,>(p: { items: T[] }): T[] => p.items

/** Every item of a paged listing, following ``next_cursor`` to the last page. */
const pages = async <T,>(list: (cursor?: string) => Promise<{ items: T[]; next_cursor?: string | null }>): Promise<T[]> => {
  const first = await list()
  const items = [...first.items]
  let cursor = first.next_cursor
  while (cursor) {
    const next = await list(cursor)
    items.push(...next.items)
    cursor = next.next_cursor
  }
  return items
}

export const useStore = create<Store>((set, get) => ({
  computes: [],
  history: [],
  nodes: {},
  tasks: {},
  providers: [],
  providerKinds: [],
  offers: [],
  events: [],
  windows: {},
  feed: null,
  functions: {},

  sel: null,
  metric: 'gpu',
  shell: false,
  logFollow: true,
  act: { kind: 'logs', compute: 'all', rank: 'all', level: 'all', q: '', since: 'all' },
  hist: { q: '', cause: 'all', provider: 'all', accel: 'all', since: 'all' },
  task: { compute: 'all', state: 'all' },
  market: { accel: 'all', market: 'all', sort: 'price' },
  sheet: null,

  metrics: {},
  hm: {},
  raw: {},
  progress: {},
  bands: {},
  costs: {},
  loading: false,
  error: null,
  taskFeed: null,
  histPages: null,
  offerPages: null,

  setUi: (patch) => set(patch),
  setEntities: (patch) => set(patch),
  pick: (sel) => set({ sel }),
  openSheet: (sheet) => set({ sheet }),
  closeSheet: () => set({ sheet: null }),

  apply: (event) => enqueue(event),

  load: async () => {
    set({ loading: true, error: null })
    try {
      const [computes, ended, providers, providerKinds] = await Promise.all([
        pages((cursor) => api.computes({ live: true, cursor })).then(alive),
        api.computes({ state: 'deleted', limit: HISTORY_PAGE, cause: causeOf(get().hist) }),
        api.providers().then(page),
        api.providerKinds(),
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
      set((s) => ({
        computes,
        history: retired(ended.items),
        histPages: { key: get().hist.cause, cursor: ended.next_cursor ?? null, loading: false, total: ended.total ?? null },
        nodes: { ...s.nodes, ...nodes },
        tasks: { ...s.tasks, ...Object.fromEntries(Object.entries(tasks).map(([id, ts]) => [id, merged(s.tasks[id] ?? [], ts)])) },
        feed: Object.entries(nodes).reduce((feed, [id, ns]) => ranked(feed, id, ns), s.feed),
        providers,
        providerKinds,
        loading: false,
      }))
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
        feed: ranked(s.feed, computeId, ns),
      }))
      void named(ts)
    } catch {
      /* a compute that no longer answers is dropped by the next full load */
    }
  },

  reloadHistory: async (computeId) => {
    try {
      const [ns, ts] = await Promise.all([api.nodes(computeId, { include_terminal: true }).then(page), api.tasks({ compute: computeId }).then(page)])
      set((s) => ({
        nodes: { ...s.nodes, [computeId]: ns.sort((a, b) => a.rank - b.rank) },
        tasks: { ...s.tasks, [computeId]: merged(s.tasks[computeId] ?? [], ts, Infinity) },
        feed: ranked(s.feed, computeId, ns),
      }))
      void named(ts)
    } catch {
      /* what the daemon no longer keeps is simply not shown */
    }
  },

  learn: async (ids) => {
    const state = get()
    const known = new Set([...state.computes.map((c) => c.id), ...state.history.map((c) => c.id)])
    const wanted = [...new Set(ids)].filter((id) => !known.has(id) && !learning.has(id))
    if (!wanted.length) return
    for (const id of wanted) learning.add(id)
    const found = (await Promise.all(wanted.map((id) => api.compute(id).catch(() => null)))).filter((c): c is Compute => c !== null)
    for (const id of wanted) learning.delete(id)
    if (!found.length) return
    set((s) => ({
      computes: [...s.computes, ...found.filter((c) => c.status.state !== 'deleted' && !s.computes.some((x) => x.id === c.id))],
      history: retired([...found.filter((c) => c.status.state === 'deleted'), ...s.history.filter((c) => !found.some((f) => f.id === c.id))]),
    }))
  },

  pageLogs: async () => {
    const feed = get().feed
    if (!feed || feed.loading || !feed.cursor) return
    set({ feed: { ...feed, loading: true } })
    try {
      const read = await api.log({ ...asked(feed.scope), limit: CONSOLE_PAGE, cursor: feed.cursor })
      const older = read.items.flatMap((entry) => lined(get(), recorded(entry)))
      set((s) => {
        if (s.feed?.key !== feed.key) return {}
        const held = s.feed.held + older.length
        return { feed: { ...s.feed, lines: spliced(s.feed.lines, older, held), cursor: read.next_cursor ?? null, loading: false, held } }
      })
    } catch {
      set((s) => (s.feed?.key === feed.key ? { feed: { ...s.feed, loading: false } } : {}))
    }
  },

  pageEvents: async (computeId, kind) => {
    const key = windowKey(computeId, kind)
    const at = get().windows[key]
    if (at && (at.loading || !at.cursor)) return
    const holding: EventWindow = { items: at?.items ?? [], cursor: at?.cursor ?? null, loading: true }
    set((s) => ({ windows: { ...s.windows, [key]: holding } }))
    try {
      const read = await api.log({ compute: computeId ?? undefined, types: framesOf(kind), limit: EVENTS_PAGE, cursor: at?.cursor ?? undefined })
      set((s) => ({
        windows: {
          ...kept(s.windows, key),
          [key]: { items: [...(s.windows[key]?.items ?? []), ...read.items.map(recorded)], cursor: read.next_cursor ?? null, loading: false },
        },
      }))
    } catch {
      set((s) => ({ windows: { ...s.windows, [key]: { ...holding, loading: false } } }))
    }
  },

  pageTasks: async (reset = false) => {
    const key = taskKey(get().task)
    const at = get().taskFeed
    const fresh = reset || at?.key !== key
    if (!fresh && (at.loading || !at.cursor)) return
    set({ taskFeed: { key, items: fresh ? [] : at.items, cursor: fresh ? null : at.cursor, loading: true, total: fresh ? null : at.total } })
    try {
      const filters = get().task
      const read = await api.tasks({
        limit: TASKS_PAGE,
        compute: filters.compute === 'all' ? undefined : filters.compute,
        state: filters.state === 'all' ? undefined : filters.state,
        cursor: fresh ? undefined : (at?.cursor ?? undefined),
      })
      set((s) => ({
        tasks: filed(s.tasks, read.items),
        taskFeed: {
          key,
          items: [...(s.taskFeed?.key === key && !fresh ? s.taskFeed.items : []), ...read.items],
          cursor: read.next_cursor ?? null,
          loading: false,
          total: read.total ?? null,
        },
      }))
      void named(read.items)
      void get().learn(read.items.map((t) => t.compute_id))
    } catch {
      set({ taskFeed: at })
    }
  },

  pageHistory: async (reset = false) => {
    const cause = get().hist.cause
    const at = get().histPages
    const fresh = reset || at?.key !== cause
    if (!fresh && (at.loading || !at.cursor)) return
    set({ histPages: { key: cause, cursor: fresh ? null : at.cursor, loading: true, total: fresh ? null : at.total } })
    try {
      const read = await api.computes({ state: 'deleted', limit: HISTORY_PAGE, cause: causeOf(get().hist), cursor: fresh ? undefined : (at?.cursor ?? undefined) })
      set((s) => ({
        history: retired(fresh ? read.items : [...s.history, ...read.items.filter((c) => !s.history.some((x) => x.id === c.id))]),
        histPages: { key: cause, cursor: read.next_cursor ?? null, loading: false, total: read.total ?? null },
      }))
    } catch {
      set({ histPages: at })
    }
  },

  moreOffers: async () => {
    const at = get().offerPages
    if (!at || at.loading || (at.total !== null && at.limit >= at.total)) return
    set({ offerPages: { ...at, limit: at.limit + OFFERS_PAGE } })
    await get().reloadOffers()
  },

  reloadProviders: async () => {
    const [providers, providerKinds] = await Promise.all([api.providers().then(page), api.providerKinds()])
    set({ providers, providerKinds })
  },

  reloadOffers: async () => {
    const f = get().market
    const key = catalogKey(f)
    const at = get().offerPages
    const limit = at?.key === key ? at.limit : OFFERS_PAGE
    set({ offerPages: { key, limit, loading: true, total: at?.key === key ? at.total : null } })
    try {
      const read = await api.offers({ accelerator: f.accel === 'all' ? undefined : f.accel, spot: f.market === 'spot' ? true : undefined, sort: f.sort, limit })
      set({ offers: read.items, offerPages: { key, limit, loading: false, total: read.total ?? null } })
    } catch {
      set({ offerPages: { key, limit, loading: false, total: at?.key === key ? at.total : null } })
    }
  },

  live: () => subscribe(enqueue, { types: STATE_FRAMES, lastEventId: HEAD }),
}))

/* ---------- ingestion ---------- */

type Patch = Partial<Store>

const RELOAD_EVERY = 2000
const HIDDEN_EVERY = 100
const GAUGES_EVERY = 1000
const TASKS_MAX = 500
const TASKS_PAGE = 100

const queue: SkyEvent[] = []
let frame: number | null = null
let timer: ReturnType<typeof setTimeout> | null = null
const gauges: SkyEvent[] = []
let gaugeTimer: ReturnType<typeof setTimeout> | null = null

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
  const printedLines: LogLine[] = []

  for (const event of batch) {
    const payload = event.data
    /* a line is kept before the store knows its compute: a window read ahead of the first load is not lost */
    if (payload.type === 'node.console') {
      if (draft.feed && inScope(draft.feed.scope, payload)) printedLines.push(...printed(draft, event, payload.compute, payload.node, payload.content))
      continue
    }
    const compute = event.compute
    if (compute && !known.has(compute)) {
      if (event.type.startsWith('compute.')) stale.add(compute)
      continue
    }
    const folded = fold(draft, event)
    if (folded) {
      Object.assign(draft, folded)
      Object.assign(patch, folded)
    }
    if (compute && live.has(compute) && (event.type === 'node.state' || event.type === 'task.state' || (event.type.startsWith('compute.') && event.type !== 'compute.cost'))) stale.add(compute)
  }

  if (printedLines.length && draft.feed) {
    const feed: LogFeed = { ...draft.feed, lines: spliced(draft.feed.lines, printedLines, draft.feed.held) }
    draft.feed = feed
    patch.feed = feed
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
    case 'compute.cost':
      return { costs: { ...s.costs, [payload.compute]: payload.cost } }
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

/**
 * The live event log, newest first, capped — never re-sorted.
 *
 * Only what the daemon records goes in: a published frame carries no sequence of its own
 * to be told apart by, and a bootstrap phase is read with the console.
 */
const logged = (s: Store, event: SkyEvent): Patch => ({ events: [event, ...s.events].slice(0, EVENTS_MAX) })

const printed = (s: Store, event: SkyEvent, computeId: string, nodeId: string, content: string): LogLine[] => {
  const rank = rankOf(s, computeId, nodeId)
  const sequence = Number(event.id)
  return content
    .split('\n')
    .filter(Boolean)
    .map<LogLine>((text, part) => ({ compute: computeId, sequence, part, node: nodeId, rank, at: event.at, level: levelOf(text), text }))
}

/** The daemon's order of printed lines: by the event that carried them, then down the event. */
export const lineOrder = (a: LogLine, b: LogLine): number => a.sequence - b.sequence || a.part - b.part

/**
 * The feed's lines with a batch laid in: each line once, in the daemon's order, the newest kept.
 *
 * The live tail only appends, and what it appends is capped — a compute printing a
 * thousand lines a second would otherwise be the page's memory. A page read back from
 * the log can overlap the lines already held, or land behind them, and then the list is
 * put back in order and the repeats dropped; ``held`` is how many of them were asked for
 * that way, and the cap makes room for all of them.
 */
function spliced(known: readonly LogLine[], fresh: readonly LogLine[], held = 0): LogLine[] {
  const lines = [...known, ...fresh]
  const ordered = lines.every((l, i) => i === 0 || lineOrder(lines[i - 1], l) < 0)
  return (ordered ? lines : lines.sort(lineOrder).filter((l, i, sorted) => i === 0 || lineOrder(sorted[i - 1], l) !== 0)).slice(-(LOGS_MAX + held))
}

/** Rank the lines a compute printed before the store had the rows of the nodes that printed them. */
function ranked(feed: LogFeed | null, computeId: string, nodes: readonly Node[]): LogFeed | null {
  const ranks = new Map(nodes.map((n) => [n.id, n.rank] as const))
  if (!feed?.lines.some((l) => l.rank === null && l.compute === computeId && ranks.has(l.node))) return feed
  return { ...feed, lines: feed.lines.map((l) => (l.rank === null && l.compute === computeId ? { ...l, rank: ranks.get(l.node) ?? null } : l)) }
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
 * is newest first by submission and holds at most ``max``: a live compute's reload keeps
 * ``TASKS_MAX``, while the older pages someone asked for, and what an ended compute left
 * behind, are kept whole.
 */
function merged(known: readonly Task[], latest: readonly Task[], max = TASKS_MAX): Task[] {
  const byId = new Map(known.map((t) => [t.id, t] as const))
  for (const t of latest) byId.set(t.id, t)
  return [...byId.values()].sort((a, b) => ms(b.submitted_at) - ms(a.submitted_at)).slice(0, max)
}

/** A page of tasks from every compute, each laid over what its own compute already has. */
function filed(known: Record<string, Task[]>, listed: readonly Task[]): Record<string, Task[]> {
  const tasks = { ...known }
  for (const id of new Set(listed.map((t) => t.compute_id))) tasks[id] = merged(tasks[id] ?? [], listed.filter((t) => t.compute_id === id), Infinity)
  return tasks
}

/* ---------- function names ---------- */

const asking = new Set<string>()
const learning = new Set<string>()

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

/* ---------- what the log views listen to ---------- */

const CONSOLE_PAGE = 600
const EVENTS_PAGE = 200
const WINDOWS_MAX = 8
const NO_EVENTS: SkyEvent[] = []

const scopeKey = (scope: LogScope): string => `${scope.compute ?? ''}|${scope.node ?? ''}|${(scope.contains ?? []).join('\u0000')}`

const taskKey = (f: TaskFilters): string => `${f.compute}|${f.state}`

const catalogKey = (f: MarketFilters): string => `${f.accel}|${f.market}|${f.sort}`

const causeOf = (f: HistoryFilters): Ending['cause'] | undefined => (f.cause === 'all' ? undefined : f.cause)

/** What a scope asks the log for. Every filter is the daemon's: a page filtered here is a page of somebody else's lines. */
const asked = (scope: LogScope): LogQuery => ({
  compute: scope.compute,
  node: scope.node,
  contains: scope.contains?.length ? scope.contains : undefined,
  types: CONSOLE_FRAMES,
})

/** The lines one event printed, and none for an event that printed nothing. */
const lined = (s: Store, event: SkyEvent): LogLine[] =>
  event.data.type === 'node.console' ? printed(s, event, event.data.compute, event.data.node, event.data.content) : []

/**
 * Whether a line the stream carried belongs to the scope its feed was read under.
 *
 * The stream narrows by compute and frame and by nothing else, so a feed scoped to one
 * node or to a search has to keep its own door: what the daemon cannot filter on the way
 * out is filtered on the way in, rather than landing in a page of somebody else's lines.
 */
const inScope = (scope: LogScope, line: { compute: string; node: string; content: string }): boolean =>
  (!scope.compute || line.compute === scope.compute) &&
  (!scope.node || line.node === scope.node) &&
  (!scope.contains?.length || scope.contains.some((text) => line.content.toLowerCase().includes(text.toLowerCase())))

/**
 * Where the live tail opens.
 *
 * Past the newest entry of the page for a scope the stream can follow, so nothing falls
 * between the two. On the head for a scope it cannot: the newest line that matched a
 * search can be months old, and resuming the stream from it would replay every line
 * printed since. A search is a read of the past, and only the matches after it are news.
 */
const tailFrom = (scope: LogScope, read: Page<LogEntry>): string =>
  !scope.node && !scope.contains?.length && read.items.length ? String(read.items[0].sequence) : HEAD

const framesOf = (kind: string): readonly string[] => (kind === 'all' ? STATE_FRAMES : STATE_FRAMES.filter((frame) => frame.startsWith(`${kind}.`)))

const windowKey = (computeId: string | null, kind: string): string => `${computeId ?? ''}|${kind}`

/** The windows worth keeping, with room made for one more. */
const kept = (windows: Record<string, EventWindow>, key: string): Record<string, EventWindow> =>
  Object.fromEntries(
    Object.entries(windows)
      .filter(([held]) => held !== key)
      .slice(1 - WINDOWS_MAX),
  )

let followed: (Subscription & { key: string }) | null = null

/**
 * Follow one scope's console: a compute or every compute, one node or all of them, filtered as asked.
 *
 * The newest page comes off the log newest first and is folded oldest first, the way the
 * stream's lines are, each stamped with when it was recorded. The stream then opens past
 * the newest of them, so nothing falls between the two and nothing is sent twice; a page
 * the log cannot give is an empty one, and the stream opens on the head.
 *
 * Console output is the bulk of the log — a hundred thousand lines of it — so one scope
 * is followed at a time, and opening another closes this one.
 */
function follow(scope: LogScope, key: string): void {
  if (followed?.key === key) return
  followed?.close()
  let tail: Subscription | null = null
  let closed = false
  followed = {
    key,
    close: () => {
      closed = true
      tail?.close()
    },
  }
  useStore.setState({ feed: { key, scope, lines: [], cursor: null, loading: true, held: 0 } })
  void api
    .log({ ...asked(scope), limit: CONSOLE_PAGE })
    .catch((): Page<LogEntry> => ({ items: [], next_cursor: null }))
    .then((read) => {
      if (closed) return
      for (const entry of [...read.items].reverse()) enqueue(recorded(entry))
      useStore.setState((s) => (s.feed?.key === key ? { feed: { ...s.feed, cursor: read.next_cursor ?? null, loading: false } } : {}))
      tail = subscribe(enqueue, { compute: scope.compute, types: CONSOLE_FRAMES, lastEventId: tailFrom(scope, read) })
    })
}

/** The console a view shows, and how far back it has been read: one compute or every one, live or ended. */
export function useLogs(scope: LogScope): LogFeed | null {
  const key = scopeKey(scope)
  useEffect(() => {
    follow(scope, key)
  }, [key])
  return useStore((s) => (s.feed?.key === key ? s.feed : null))
}

/**
 * A scope's events, newest first: what the log has been read back to, and every live frame since.
 *
 * The live slice is capped for the whole daemon, so the window is kept beside it rather
 * than folded in, and the two are joined by sequence where they overlap. ``kind`` is the
 * daemon's filter too — paging a list somebody then filters here pages somebody else's events.
 */
export function useEvents(computeId: string | null, kind = 'all'): EventWindow {
  const key = windowKey(computeId, kind)
  const held = useStore((s) => s.windows[key])
  const live = useStore((s) => s.events)
  useEffect(() => {
    if (!useStore.getState().windows[key]) void useStore.getState().pageEvents(computeId, kind)
  }, [key, computeId, kind])
  return useMemo(() => {
    const frames = framesOf(kind)
    const byId = new Map<string, SkyEvent>()
    for (const e of live) if ((!computeId || e.compute === computeId) && frames.includes(e.frame)) byId.set(e.id, e)
    for (const e of held?.items ?? NO_EVENTS) byId.set(e.id, e)
    return { items: [...byId.values()].sort((a, b) => Number(b.id) - Number(a.id)), cursor: held?.cursor ?? null, loading: held?.loading ?? true }
  }, [computeId, kind, held, live])
}

/** The tasks the Tasks view is looking at, its first page read when the filters change. */
export function useTasks(): TaskFeed | null {
  const key = useStore((s) => taskKey(s.task))
  const feed = useStore((s) => s.taskFeed)
  useEffect(() => {
    if (useStore.getState().taskFeed?.key !== key) void useStore.getState().pageTasks(true)
  }, [key])
  return feed?.key === key ? feed : null
}

/** Where the history listing stands, its first page read again when the cause asked for changes. */
export function useHistory(): Pages | null {
  const key = useStore((s) => s.hist.cause)
  const pages = useStore((s) => s.histPages)
  useEffect(() => {
    if (useStore.getState().histPages?.key !== key) void useStore.getState().pageHistory(true)
  }, [key])
  return pages?.key === key ? pages : null
}

/** Where the catalog listing stands, read again when the market's filters or its order change. */
export function useOffers(): Catalog | null {
  const key = useStore((s) => catalogKey(s.market))
  const pages = useStore((s) => s.offerPages)
  useEffect(() => {
    if (useStore.getState().offerPages?.key !== key) void useStore.getState().reloadOffers()
  }, [key])
  return pages?.key === key ? pages : null
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
  return patch.gpu === undefined ? folded : { ...folded, bands: band({ ...s, metrics }, computeId) }
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
  if (!values.length) return s.bands
  const point: BandPoint = { min: Math.min(...values), med: median(values), max: Math.max(...values) }
  return { ...s.bands, [computeId]: [...(s.bands[computeId] ?? seedBand(point)), point].slice(-BAND) }
}

/* ---------- what the store keeps ---------- */

/** A deleted compute is not part of the fleet: no card, no total, no metrics feed. */
const alive = (computes: readonly Compute[]): Compute[] => computes.filter((c) => c.status.state !== 'deleted')

/** History, the most recently ended first. */
const retired = (computes: readonly Compute[]): Compute[] => [...computes].sort((a, b) => endedAt(b) - endedAt(a))

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

/** The freshest copy of a task: the one a reload of its compute brought in, else the one the page carried. */
export const freshest = (state: Store, task: Task): Task => (state.tasks[task.compute_id] ?? []).find((t) => t.id === task.id) ?? task

export const progressOf = (state: Store, nodeId: string): NodeProgress | undefined => state.progress[nodeId]

/** A band opens on a full window, drifting around the first reading, as the prototype's does. */
const seedBand = (p: BandPoint): BandPoint[] =>
  Array.from({ length: BAND }, (_, i) => {
    const drift = Math.sin(i / 6) * 5
    return { min: clamp(p.min + drift - 2, 0, 100), med: clamp(p.med + drift, 0, 100), max: clamp(p.max + drift, 0, 100) }
  })

export const bandOf = (state: Store, computeId: string): BandPoint[] => state.bands[computeId] ?? []

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

/** What a compute has cost: its closing bill once it has ended, the daemon's last reading while it runs. */
export const spentOf = (state: Store, c: Compute): number | undefined => c.ended?.cost ?? state.costs[c.id]
