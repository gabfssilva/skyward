import { useEffect, useRef, type ReactNode } from 'react'
import type { SkyEvent } from '../api/events'
import type { LogLine } from '../state/store'
import { DAY, clock, dateOf } from '../state/model'

/**
 * A window on what the machines printed: oldest line first, the newest last — the order it was written in.
 *
 * The box follows the tail while it is asked to, and what came before the window is loaded above it,
 * because that is the end the older lines belong to.
 */
export function LogBox({
  lines,
  names,
  ranks = true,
  tall,
  follow = true,
  empty = 'Nothing printed yet.',
  older,
}: {
  lines: readonly LogLine[]
  names?: ReadonlyMap<string, string>
  ranks?: boolean
  tall?: boolean
  follow?: boolean
  empty?: string
  older?: ReactNode
}) {
  const box = useRef<HTMLDivElement>(null)
  const last = lines[lines.length - 1]
  useEffect(() => {
    if (follow && box.current) box.current.scrollTop = box.current.scrollHeight
  }, [last?.sequence, last?.part, follow])
  return (
    <>
      {older}
      <div className={['logbox', tall ? 'tall' : null, ranks ? null : 'norank'].filter(Boolean).join(' ')} ref={box}>
        {lines.length ? lines.map((l) => <LogLineRow key={`${l.sequence}.${l.part}`} line={l} computeName={names?.get(l.compute)} />) : <div className="sub">{empty}</div>}
      </div>
    </>
  )
}

/** A window on what the daemon recorded, read the way the console beside it is: oldest first. */
export function EvBox({
  events,
  names,
  tall,
  empty = 'Nothing happened yet.',
  older,
}: {
  events: readonly SkyEvent[]
  names?: ReadonlyMap<string, string>
  tall?: boolean
  empty?: string
  older?: ReactNode
}) {
  const box = useRef<HTMLDivElement>(null)
  const newest = events[0]
  useEffect(() => {
    if (box.current) box.current.scrollTop = box.current.scrollHeight
  }, [newest?.id])
  return (
    <>
      {older}
      <div className={tall ? 'evbox tall' : 'evbox'} ref={box}>
        {events.length ? (
          events
            .slice()
            .reverse()
            .map((e) => <EvLineRow key={e.id} event={e} where={e.compute ? names?.get(e.compute) : undefined} />)
        ) : (
          <div className="sub">{empty}</div>
        )}
      </div>
    </>
  )
}

/** One printed line, the way the prototype's ``logLine`` reads: time, ``compute/rank`` — no rank while it is unknown — text. */
export function LogLineRow({ line, computeName }: { line: LogLine; computeName?: string }) {
  return (
    <div className={`logline${line.level === 'err' ? ' err' : line.level === 'warn' ? ' warn' : ''}`}>
      <span>{clock(line.at)}</span>
      <span className="rk">{computeName ? <>{computeName}{line.rank === null ? null : <i>/{line.rank}</i>}</> : line.rank === null ? '' : `rank ${line.rank}`}</span>
      <div>{line.text}</div>
    </div>
  )
}

/** What colour an event type is printed in: tasks in the accent, trouble red, computes green, the rest blue. */
export const evTint = (type: string): string =>
  type.startsWith('task') ? 'var(--accent)' : /degraded|failed|lost/.test(type) ? 'var(--bad)' : type.startsWith('compute') ? 'var(--ok)' : 'var(--boot)'

/**
 * One event: time (the date when older than a day), tinted type, text, and where.
 *
 * ``where`` is left out on a page that is already about one compute — every line there ended in the
 * name of the compute whose page it was.
 */
export function EvLineRow({ event, where }: { event: SkyEvent; where?: string }) {
  return (
    <div className="evline">
      <span className="mono faint" style={{ whiteSpace: 'nowrap' }}>
        {Date.now() - event.at > DAY ? dateOf(event.at) : clock(event.at)}
      </span>
      <span className="mono" style={{ color: evTint(event.type) }}>
        {event.type}
      </span>
      <div className="trunc">
        {event.text}
        {where ? (
          <span className="faint">
            {' — '}
            {where}
            {event.node ? ` ${event.node}` : ''}
          </span>
        ) : null}
      </div>
    </div>
  )
}
