/**
 * The daemon's own log, as the Daemon tab of Activity reads it.
 *
 * A window is summarized by the daemon — counted by level on sixty steps, by component and by group — and
 * then followed: the stream opens from the sequence that summary counted up to, and every line it carries
 * is added to the counts here, so a line is neither counted twice nor missed. The window is read again each
 * time it slides by a step, and the lines that arrived meanwhile are added again on top of what came back.
 *
 * The lines themselves are read only when somebody looks at them, the newest page first, and the same
 * stream appends to them.
 */
import { useEffect, useMemo, useRef, useState } from 'react'
import { api, ApiError } from '../api/client'
import type { DaemonGroup, DaemonLine, DaemonLogQuery, DaemonSummary, LogLevel } from '../api/client'
import { listen } from '../api/events'

export type Window = '15m' | '1h' | '6h' | '24h'

/** What the tab is looking at. ``range`` narrows the window to a stretch picked on the chart; ``group`` is one group's lines. */
export type DaemonFilters = {
  level: LogLevel
  compute: 'all' | string
  node: string | null
  components: readonly string[]
  group: string | null
  hidden: readonly string[]
  q: string
  window: Window
  range: readonly [number, number] | null
  view: 'grouped' | 'lines'
}

export const DAEMON: DaemonFilters = {
  level: 'DEBUG',
  compute: 'all',
  node: null,
  components: [],
  group: null,
  hidden: [],
  q: '',
  window: '1h',
  range: null,
  view: 'grouped',
}

export const WINDOWS: readonly (readonly [Window, string])[] = [
  ['15m', '15m'],
  ['1h', '1h'],
  ['6h', '6h'],
  ['24h', '24h'],
]

export const LEVELS: readonly LogLevel[] = ['DEBUG', 'INFO', 'WARNING', 'ERROR']

/** How many steps a window is counted in: the chart's columns. */
export const STEPS = 60

/** How a level is drawn and named: the same four everywhere a line is shown. */
export const TINT: Record<LogLevel, 'debug' | 'info' | 'warn' | 'err'> = { DEBUG: 'debug', INFO: 'info', WARNING: 'warn', ERROR: 'err' }

export const severity = (level: LogLevel): number => LEVELS.indexOf(level)

/** Most severe first, then most frequent: the order the daemon hands groups over in, kept as lines are added. */
export const bySeverity = (a: DaemonGroup, b: DaemonGroup): number => severity(b.level) - severity(a.level) || b.count - a.count

/** The window a summary covers: sixty steps ending with the one now falls in, so a window read again lines up with the last. */
export type Span = { since: number; until: number; step: number }

export const spanOf = (window: Window, now: number): Span => {
  const step = WIDTH[window] / STEPS
  const until = (Math.floor(now / step) + 1) * step
  return { since: until - WIDTH[window], until, step }
}

/** The filters as the daemon takes them, the window aside. */
export const asked = (f: DaemonFilters): DaemonLogQuery => ({
  level: f.level === 'DEBUG' ? undefined : f.level,
  compute: f.compute === 'all' ? undefined : f.compute,
  node: f.node ?? undefined,
  component: f.components.length ? f.components : undefined,
  group: f.group ? [f.group] : undefined,
  hide: f.hidden.length ? f.hidden : undefined,
  contains: f.q.trim() ? [f.q.trim()] : undefined,
})

/** A summary with one more line counted in, unless it was counted already. */
export function fold(summary: DaemonSummary, line: DaemonLine): DaemonSummary {
  if (line.sequence <= summary.sequence) return summary
  const { since, step } = summary.volume
  const steps = summary.volume.debug.length
  const index = Math.floor((Date.parse(line.at) - since) / step)
  if (index < 0 || index >= steps) return { ...summary, sequence: line.sequence }
  const column = VOLUME[line.level]
  const found = summary.groups.find((g) => g.key === line.group)
  const group: DaemonGroup = found
    ? {
        ...found,
        count: found.count + 1,
        last: line.at,
        latest: line,
        level: severity(line.level) > severity(found.level) ? line.level : found.level,
        series: bumped(found.series, index),
      }
    : {
        key: line.group,
        site: line.site,
        exception: line.exception?.type ?? null,
        component: line.component,
        level: line.level,
        count: 1,
        first: line.at,
        last: line.at,
        computes: line.compute ? 1 : 0,
        series: bumped(new Array<number>(steps).fill(0), index),
        latest: line,
      }
  return {
    sequence: line.sequence,
    volume: { ...summary.volume, [column]: bumped(summary.volume[column], index) },
    components: line.component ? { ...summary.components, [line.component]: (summary.components[line.component] ?? 0) + 1 } : summary.components,
    groups: [...summary.groups.filter((g) => g !== found), group].sort(bySeverity),
  }
}

/** What the tab draws: the window's summary, the groups of the stretch it is narrowed to, and the lines, oldest first. */
export type DaemonLog = {
  summary: DaemonSummary | null
  groups: readonly DaemonGroup[] | null
  lines: readonly DaemonLine[] | null
  older: boolean
  loadingOlder: boolean
  loadOlder: () => void
  /** a daemon that keeps no log: one embedded in another process */
  missing: boolean
  error: string | null
}

export function useDaemonLog(f: DaemonFilters): DaemonLog {
  const query = useMemo(() => asked(f), [f.level, f.compute, f.node, f.components, f.group, f.hidden, f.q])
  const key = JSON.stringify(query)
  const [now, setNow] = useState(() => Date.now())
  const span = useMemo(() => spanOf(f.window, now), [f.window, now])
  const [summary, setSummary] = useState<Keyed<DaemonSummary> | null>(null)
  const [origin, setOrigin] = useState<Keyed<number> | null>(null)
  const [ranged, setRanged] = useState<Keyed<DaemonSummary> | null>(null)
  const [lines, setLines] = useState<Held | null>(null)
  const [failure, setFailure] = useState<ApiError | Error | null>(null)
  const heard = useRef<Keyed<DaemonLine[]>>({ key: '', value: [] })

  useEffect(() => {
    const slide = setTimeout(() => setNow(Date.now()), Math.max(1000, span.until - Date.now()))
    return () => clearTimeout(slide)
  }, [span.until])

  useEffect(() => {
    let live = true
    api.daemonLogSummary({ ...query, since: span.since, until: span.until, step: span.step }).then(
      (read) => {
        if (!live) return
        setSummary({ key, value: (heard.current.key === key ? heard.current.value : []).reduce(fold, read) })
        setOrigin((held) => (held?.key === key ? held : { key, value: read.sequence }))
        setFailure(null)
      },
      (error: unknown) => live && setFailure(error instanceof Error ? error : new Error(String(error))),
    )
    return () => {
      live = false
    }
  }, [key, span.since, span.until, span.step])

  const range = f.range
  useEffect(() => {
    if (!range) return setRanged(null)
    let live = true
    const [since, until] = range
    api.daemonLogSummary({ ...query, since, until, step: Math.max(1, Math.ceil((until - since) / STEPS)) }).then(
      (read) => live && setRanged({ key, value: read }),
      () => undefined,
    )
    return () => {
      live = false
    }
  }, [key, range])

  const bounds = useMemo(() => (range ? { since: range[0], until: range[1] } : { since: spanOf(f.window, Date.now()).since }), [range, f.window])
  const reading = f.view === 'lines'
  useEffect(() => {
    if (!reading) return setLines(null)
    let live = true
    api.daemonLog({ ...query, ...bounds, limit: PAGE }).then(
      (page) => live && setLines({ key, items: page.items.slice().reverse(), cursor: page.next_cursor ?? null, loading: false }),
      () => undefined,
    )
    return () => {
      live = false
    }
  }, [key, reading, bounds])

  const ranging = useRef(range)
  ranging.current = range
  const start = origin?.key === key ? origin.value : null
  useEffect(() => {
    if (start === null) return
    heard.current = { key, value: [] }
    let pending: DaemonLine[] = []
    let frame = 0
    const flush = () => {
      frame = 0
      const arrived = pending
      pending = []
      setSummary((held) => (held?.key === key ? { key, value: arrived.reduce(fold, held.value) } : held))
      if (!ranging.current) setLines((held) => (held?.key === key ? appended(held, arrived) : held))
    }
    const subscription = listen(
      api.daemonLogStream(query),
      (message) => {
        if (message.frame !== 'log') return
        const line = JSON.parse(message.data) as DaemonLine
        heard.current = { key, value: [...heard.current.value.slice(-HEARD), line] }
        pending.push(line)
        if (!frame) frame = requestAnimationFrame(flush)
      },
      String(start),
    )
    return () => {
      subscription.close()
      if (frame) cancelAnimationFrame(frame)
    }
  }, [key, start])

  const held = lines?.key === key ? lines : null
  const loadOlder = () => {
    if (!held || held.loading || !held.cursor) return
    setLines({ ...held, loading: true })
    api.daemonLog({ ...query, ...bounds, cursor: held.cursor, limit: PAGE }).then(
      (page) =>
        setLines((latest) =>
          latest?.key === key ? { key, items: [...page.items.slice().reverse(), ...latest.items], cursor: page.next_cursor ?? null, loading: false } : latest,
        ),
      () => setLines((latest) => (latest?.key === key ? { ...latest, loading: false } : latest)),
    )
  }

  const current = summary?.key === key ? summary.value : null
  return {
    summary: current,
    groups: range ? (ranged?.key === key ? ranged.value.groups : null) : (current?.groups ?? null),
    lines: held?.items ?? null,
    older: held?.cursor != null,
    loadingOlder: held?.loading ?? false,
    loadOlder,
    missing: failure instanceof ApiError && failure.status === 404,
    error: failure && !(failure instanceof ApiError && failure.status === 404) ? failure.message : null,
  }
}

const WIDTH: Record<Window, number> = { '15m': 9e5, '1h': 3.6e6, '6h': 2.16e7, '24h': 8.64e7 }
const VOLUME = { DEBUG: 'debug', INFO: 'info', WARNING: 'warning', ERROR: 'error' } as const
const PAGE = 200
const LINES_MAX = 2000
/** The live lines kept to be added again onto a summary read while they were arriving. */
const HEARD = 5000

type Keyed<T> = { key: string; value: T }
type Held = { key: string; items: DaemonLine[]; cursor: string | null; loading: boolean }

const bumped = (counts: readonly number[], index: number): number[] => counts.map((n, i) => (i === index ? n + 1 : n))

/** The lines a page holds with the ones that arrived after its newest, the oldest let go past the cap. */
function appended(held: Held, arrived: readonly DaemonLine[]): Held {
  const newest = held.items[held.items.length - 1]?.sequence ?? 0
  const fresh = arrived.filter((line) => line.sequence > newest)
  if (!fresh.length) return held
  const items = [...held.items, ...fresh]
  return { ...held, items: items.length > LINES_MAX ? items.slice(-LINES_MAX) : items }
}
