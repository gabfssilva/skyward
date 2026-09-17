import { useNavigate } from 'react-router-dom'
import type { Compute, Offer } from '../../api/client'
import { useHistory, useStore } from '../../state/store'
import type { HistoryFilters } from '../../state/store'
import { CAUSE, DAY, ago, callsOf, dur, endedAt, failedOf, money, ranOf, targetOf } from '../../state/model'
import { Pill, TableScroll } from '../../ui/primitives'

const SINCE: Record<HistoryFilters['since'], number> = { '24h': DAY, '7d': 7 * DAY, '30d': 30 * DAY, all: Infinity }

const CAUSES: readonly (readonly [HistoryFilters['cause'], string])[] = [
  ['all', 'all'],
  ['requested', 'asked for'],
  ['abandoned', 'abandoned'],
]

const boundOf = (c: Compute): Pick<Offer, 'kind' | 'accelerator' | 'accelerator_count' | 'region'> | null => {
  const s = c.spec.specs[0]
  return c.offer ?? (s ? { ...s, kind: s.provider.kind } : null)
}

const providerOf = (c: Compute): string => boundOf(c)?.kind ?? '—'
const accelOf = (c: Compute): string => boundOf(c)?.accelerator ?? '—'

/** ``64× 8× H100 · aws · us-east-1``, from the offer the compute was bound to, or the first spec it asked for; a machine with no accelerator goes by its instance type. */
const shapeOf = (c: Compute): string => {
  const b = boundOf(c)
  if (!b) return `${targetOf(c)} nodes`
  const machine = b.accelerator ? `${b.accelerator_count > 1 ? `${b.accelerator_count}× ` : ''}${b.accelerator.toUpperCase()}` : (c.offer?.instance_type ?? 'CPU')
  return `${targetOf(c)}× ${machine} · ${b.kind} · ${b.region ?? 'any'}`
}

/** The past computes that pass the filters, the most recently ended first; ``since`` is measured against the ending. */
const histList = (past: readonly Compute[], f: HistoryFilters): Compute[] => {
  const q = f.q.trim().toLowerCase()
  const now = Date.now()
  return past
    .filter(
      (c) =>
        (f.cause === 'all' || c.ended?.cause === f.cause) &&
        (f.provider === 'all' || providerOf(c) === f.provider) &&
        (f.accel === 'all' || accelOf(c) === f.accel) &&
        now - endedAt(c) <= SINCE[f.since] &&
        (!q || (c.name ?? '').toLowerCase().includes(q) || c.id.includes(q)),
    )
    .sort((a, b) => endedAt(b) - endedAt(a))
}

function Select({ k, values, value, onChange }: { k: 'provider' | 'accel'; values: readonly string[]; value: string; onChange: (v: string) => void }) {
  return (
    <select className="search" aria-label={k} value={value} onChange={(e) => onChange(e.target.value)}>
      {['all', ...values].map((v) => (
        <option key={v} value={v}>
          {v === 'all' ? `any ${k === 'accel' ? 'accelerator' : k}` : k === 'accel' ? v.toUpperCase() : v}
        </option>
      ))}
    </select>
  )
}

/** The History card: what has ended, filtered; a row opens the compute. */
export function History() {
  const navigate = useNavigate()
  const past = useStore((s) => s.history)
  const pages = useHistory()
  const pageHistory = useStore((s) => s.pageHistory)
  const f = useStore((s) => s.hist)
  const setUi = useStore((s) => s.setUi)
  const set = (patch: Partial<HistoryFilters>) => setUi({ hist: { ...f, ...patch } })
  const list = histList(past, f)
  const providers = [...new Set(past.map(providerOf))].sort()
  const accels = [...new Set(past.map(accelOf))].sort()
  const spent = list.reduce((s, c) => s + (c.ended?.cost ?? 0), 0)

  return (
    <section className="card hist">
      <div className="combhead">
        <b>History</b>
        <span className="sub">
          {list.length} of {pages?.total ?? past.length} ended · {money(spent)} across what is loaded
        </span>
      </div>
      <div className="row" style={{ gap: 8, flexWrap: 'wrap', marginBottom: 12 }}>
        <input className="search" placeholder="name or id" value={f.q} autoComplete="off" style={{ minWidth: 180 }} onChange={(e) => set({ q: e.target.value })} />
        <div className="pick">
          {CAUSES.map(([v, label]) => (
            <button key={v} aria-selected={f.cause === v} onClick={() => set({ cause: v })}>
              {label}
            </button>
          ))}
        </div>
        <Select k="provider" values={providers} value={f.provider} onChange={(provider) => set({ provider })} />
        <Select k="accel" values={accels} value={f.accel} onChange={(accel) => set({ accel })} />
        <div className="pick">
          {(Object.keys(SINCE) as HistoryFilters['since'][]).map((v) => (
            <button key={v} aria-selected={f.since === v} onClick={() => set({ since: v })}>
              {v === 'all' ? 'all time' : v}
            </button>
          ))}
        </div>
      </div>
      <TableScroll label="History">
        <table>
          <thead>
            <tr>
              <th>Compute</th>
              <th>Ended because</th>
              <th>Shape</th>
              <th>Calls</th>
              <th>Ended</th>
              <th>Ran</th>
              <th className="right">Cost</th>
            </tr>
          </thead>
          <tbody>
            {list.length ? (
              list.map((c) => {
                const ended = c.ended
                const ran = ranOf(c)
                return (
                  <tr key={c.id} style={{ cursor: 'pointer' }} onClick={() => navigate(`/computes/${c.id}`)}>
                    <td>
                      <b style={{ fontWeight: 600 }}>{c.name ?? c.id}</b>
                      <div className="mono faint">{c.id}</div>
                    </td>
                    <td>
                      <Pill state={c.status.state} />
                      <div className="why">{ended ? CAUSE[ended.cause] : '—'}</div>
                    </td>
                    <td className="sub" style={{ whiteSpace: 'nowrap' }}>
                      {shapeOf(c)}
                    </td>
                    <td className="mono" style={{ whiteSpace: 'nowrap' }}>
                      {callsOf(c) || '—'}
                      {failedOf(c) ? <span style={{ color: 'var(--bad)' }}> {failedOf(c)} failed</span> : null}
                    </td>
                    <td className="mono faint" style={{ whiteSpace: 'nowrap' }}>
                      {ended ? ago(endedAt(c)) : '—'}
                    </td>
                    <td className="mono" style={{ whiteSpace: 'nowrap' }}>
                      {ran < 60e3 ? '—' : dur(ran)}
                    </td>
                    <td className="right mono">{ended?.cost ? money(ended.cost) : '—'}</td>
                  </tr>
                )
              })
            ) : (
              <tr>
                <td colSpan={7} className="sub" style={{ padding: '22px 8px', textAlign: 'center' }}>
                  Nothing ended that matches these filters.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </TableScroll>
      {pages?.cursor ? (
        <button className="btn sm" style={{ marginTop: 10 }} disabled={pages.loading} onClick={() => void pageHistory()}>
          Show older
        </button>
      ) : null}
    </section>
  )
}
