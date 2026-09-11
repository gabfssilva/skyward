import { useStore } from '../state/store'
import { clock } from '../state/model'

const tint = (t: string): string =>
  t.startsWith('task')
    ? 'var(--accent)'
    : /degraded|failed|lost/.test(t)
      ? 'var(--bad)'
      : t.startsWith('compute')
        ? 'var(--ok)'
        : 'var(--boot)'

export function Events({ computeId }: { computeId: string | null }) {
  const events = useStore((s) => s.events)
  const computes = useStore((s) => s.computes)
  const list = (computeId ? events.filter((e) => e.compute === computeId) : events).slice(0, 60)
  return (
    <>
      {list.map((e, i) => {
        const k = computes.find((c) => c.id === e.compute)
        return (
          <div className="evline" key={`${e.id}-${i}`}>
            <span className="mono faint">{clock(e.at)}</span>
            <span className="mono" style={{ color: tint(e.type) }}>
              {e.type}
            </span>
            <div className="trunc">
              {e.text}
              <span className="faint">
                {' — '}
                {k ? (k.name ?? k.id) : e.compute}
                {e.node ? ' ' + e.node : ''}
              </span>
            </div>
          </div>
        )
      })}
    </>
  )
}
