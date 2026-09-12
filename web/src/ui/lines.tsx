import type { SkyEvent } from '../api/events'
import type { LogLine } from '../state/store'
import { DAY, clock, dateOf } from '../state/model'

/** One printed line, the way the prototype's ``logLine`` reads: time, ``compute/rank`` — no rank while it is unknown — text. */
export function LogLineRow({ line, computeName }: { line: LogLine; computeName?: string }) {
  return (
    <div className={`logline${line.level === 'err' ? ' err' : line.level === 'warn' ? ' warn' : ''}`}>
      <span>{clock(line.at)}</span>
      <span className="rk">
        {computeName ?? ''}
        {line.rank === null ? null : <i>/{line.rank}</i>}
      </span>
      <div>{line.text}</div>
    </div>
  )
}

/** What colour an event type is printed in: tasks in the accent, trouble red, computes green, the rest blue. */
export const evTint = (type: string): string =>
  type.startsWith('task') ? 'var(--accent)' : /degraded|failed|lost/.test(type) ? 'var(--bad)' : type.startsWith('compute') ? 'var(--ok)' : 'var(--boot)'

/** One event, the way the prototype's ``evLine`` reads: time (the date when older than a day), tinted type, text, and where. */
export function EvLineRow({ event, computeName }: { event: SkyEvent; computeName?: string }) {
  const where = computeName ?? event.compute ?? ''
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
