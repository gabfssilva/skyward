import { useEffect, useMemo, useRef } from 'react'
import type { SkyEvent } from '../../api/events'
import { HOUR, holderOf } from '../../state/model'
import { computeById, isLive, useEvents, useLogs, useStore, type ActivityFilters, type LogLine, type LogScope } from '../../state/store'
import { Icon } from '../../ui/icons'
import { EvLineRow, LogLineRow } from '../../ui/lines'
import { Chip, Pick } from '../../ui/primitives'

const ACT_SINCE: Record<ActivityFilters['since'], number> = { '5m': 5 * 60e3, '15m': 15 * 60e3, '1h': HOUR, all: Infinity }
const SINCE_OPTIONS = (Object.keys(ACT_SINCE) as ActivityFilters['since'][]).map((k) => [k, k === 'all' ? 'all time' : k] as const)
const LOG_LEVELS = [
  ['all', 'all'],
  ['warn', 'warnings'],
  ['err', 'errors'],
] as const
const EV_KINDS = [
  ['all', 'all'],
  ['compute', 'compute'],
  ['node', 'node'],
  ['task', 'task'],
] as const

const NONE: LogLine[] = []

/** What a level means to the daemon — the words ``levelOf`` reads a line's level from, which these have to stay in step with. */
const ERR_WORDS = ['error', 'exception', 'traceback', 'fatal'] as const
const LEVEL_WORDS: Record<string, readonly string[] | undefined> = { err: ERR_WORDS, warn: [...ERR_WORDS, 'warn'] }

const withinSince = (f: ActivityFilters, at: number): boolean => Date.now() - at <= ACT_SINCE[f.since]

const levelPass = (f: ActivityFilters, l: LogLine): boolean => f.level === 'all' || (f.level === 'warn' ? l.level !== 'info' : l.level === 'err')

const evMatch = (f: ActivityFilters, e: SkyEvent): boolean => withinSince(f, e.at) && (!f.q || `${e.text} ${e.type}`.toLowerCase().includes(f.q.toLowerCase()))

/** What a count reads as: what the daemon gave, or what a filter applied here left of it — never a total nobody said. */
const counted = (shown: number, loaded: number, noun: string): string => (shown === loaded ? `${loaded} ${noun}` : `${shown} of ${loaded} loaded`)

/** A log level is not an event kind: a pick left over from the other list asks the daemon for every kind, never for none. */
const kindOf = (level: string): string => (EV_KINDS.some(([kind]) => kind === level) ? level : 'all')

/** The Activity view: every compute's logs or the daemon's events in one card, filtered by ``Ui.act``. */
export function Stage() {
  const act = useStore((s) => s.act)
  const setUi = useStore((s) => s.setUi)
  const logFollow = useStore((s) => s.logFollow)
  const computes = useStore((s) => s.computes)
  const history = useStore((s) => s.history)
  const pageLogs = useStore((s) => s.pageLogs)
  const pageEvents = useStore((s) => s.pageEvents)
  const chosen = useStore((s) => (act.compute === 'all' ? undefined : computeById(s, act.compute)))
  const nodes = useStore((s) => (chosen ? s.nodes[chosen.id] : undefined)) ?? []
  const unloaded = useStore((s) => chosen !== undefined && !isLive(s, chosen.id) && s.nodes[chosen.id] === undefined)
  const logs = act.kind === 'logs'

  const scope: LogScope = {
    compute: act.compute === 'all' ? undefined : act.compute,
    node: act.rank === 'all' ? undefined : holderOf(nodes, act.rank)?.id,
    contains: act.q ? [act.q] : LEVEL_WORDS[act.level],
  }
  const feed = useLogs(scope)
  const events = useEvents(chosen?.id ?? null, kindOf(act.level))

  useEffect(() => {
    if (act.compute !== 'all' && !chosen) setUi({ act: { ...act, compute: 'all' } })
  }, [act, chosen, setUi])

  useEffect(() => {
    if (chosen && unloaded) void useStore.getState().reloadHistory(chosen.id)
  }, [chosen, unloaded])

  const set = (patch: Partial<ActivityFilters>) => setUi({ act: { ...act, ...patch } })

  const lines = useMemo(() => {
    const searched = act.q !== ''
    return (feed?.lines ?? NONE).filter((l) => withinSince(act, l.at) && (!searched || levelPass(act, l)))
  }, [act, feed])
  const matched = useMemo(() => events.items.filter((e) => evMatch(act, e)), [act, events])
  const names = useMemo(() => new Map([...computes, ...history].map((c) => [c.id, c.name ?? c.id])), [computes, history])

  const body = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (logs && logFollow && body.current) body.current.scrollTop = body.current.scrollHeight
  }, [lines, logs, logFollow])

  /* the log is read newest first, so once the oldest line held is past the window nothing older can match */
  const spent = feed !== null && feed.lines.length > 0 && !withinSince(act, feed.lines[0].at)
  const olderLines = feed !== null && feed.cursor !== null && !spent

  const rankOptions = [['all', 'all'] as const, ...[...new Set(nodes.map((n) => n.rank))].sort((a, b) => a - b).slice(0, 10).map((r) => [String(r), String(r)] as const)]

  return (
    <section className="card">
      <div className="row" style={{ gap: 8, flexWrap: 'wrap', marginBottom: 10 }}>
        <Pick
          value={act.kind}
          options={[
            ['logs', <><Icon name="logs" />Logs</>],
            ['events', <><Icon name="events" />Events</>],
          ]}
          onChange={(kind) => set({ kind })}
        />
        <select className="search" aria-label="compute" style={{ minWidth: 150 }} value={act.compute} onChange={(e) => set({ compute: e.target.value, rank: 'all' })}>
          <option value="all">every compute</option>
          {computes.map((c) => (
            <option key={c.id} value={c.id}>
              {c.name ?? c.id}
            </option>
          ))}
          {history.map((c) => (
            <option key={c.id} value={c.id}>
              {c.name ?? c.id} · ended
            </option>
          ))}
        </select>
        {logs && chosen ? (
          <>
            <span className="cap">rank</span>
            <Pick value={String(act.rank)} options={rankOptions} onChange={(r) => set({ rank: r === 'all' ? 'all' : Number(r) })} />
          </>
        ) : null}
        <Pick value={act.level} options={logs ? LOG_LEVELS : EV_KINDS} onChange={(level) => set({ level })} />
        <Pick value={act.since} options={SINCE_OPTIONS} onChange={(since) => set({ since })} />
        <input className="search" placeholder={logs ? 'filter lines' : 'filter events'} value={act.q} autoComplete="off" onChange={(e) => set({ q: e.target.value })} />
        <span className="mono faint" style={{ marginLeft: 'auto' }}>
          {logs ? counted(lines.length, feed?.lines.length ?? 0, 'lines') : counted(matched.length, events.items.length, 'events')}
        </span>
        {logs ? (
          <Chip pressed={logFollow} onClick={() => setUi({ logFollow: !logFollow })}>
            {logFollow ? 'following' : 'paused'}
          </Chip>
        ) : null}
      </div>
      {logs ? (
        <>
          <div className="logbox tall" ref={body}>
            {lines.length ? lines.map((l) => <LogLineRow key={`${l.sequence}.${l.part}`} line={l} computeName={names.get(l.compute)} />) : <div className="sub">No line matches.</div>}
          </div>
          {olderLines ? (
            <button className="btn sm" style={{ marginTop: 10 }} disabled={feed.loading} onClick={() => void pageLogs()}>
              Load older lines
            </button>
          ) : null}
        </>
      ) : (
        <>
          <div className="evbox" style={{ maxHeight: 'calc(100vh - 250px)' }}>
            {matched.length ? matched.map((e) => <EvLineRow key={e.id} event={e} computeName={e.compute ? names.get(e.compute) : undefined} />) : <div className="sub">No event matches.</div>}
          </div>
          {events.cursor ? (
            <button className="btn sm" style={{ marginTop: 10 }} disabled={events.loading} onClick={() => void pageEvents(chosen?.id ?? null, kindOf(act.level))}>
              Load older events
            </button>
          ) : null}
        </>
      )}
    </section>
  )
}
