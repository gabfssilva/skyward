import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import type { Compute } from '../../api/client'
import { useHistory, useStore } from '../../state/store'
import type { HistoryFilters } from '../../state/store'
import { DAY, HOUR, ago, boundOf, callsOf, dateOf, dur, endedAt, failedOf, machineOf, money, ranOf, targetOf } from '../../state/model'
import { TableScroll } from '../../ui/primitives'

const SINCE: Record<HistoryFilters['since'], number> = { '24h': DAY, '7d': 7 * DAY, '30d': 30 * DAY, all: Infinity }

const CAUSES: readonly (readonly [HistoryFilters['cause'], string])[] = [
  ['all', 'any reason'],
  ['requested', 'asked for'],
  ['abandoned', 'abandoned'],
]

const providerOf = (c: Compute): string => boundOf(c)?.kind ?? '—'
const accelOf = (c: Compute): string => boundOf(c)?.accelerator ?? '—'

/**
 * What the machine was, under the accelerator already named beside it: the instance type the provider
 * sells it as, else what it was asked to have. A provider whose instance type *is* the accelerator —
 * runpod sells an A10G as ``A10G`` — would only say it twice.
 */
const hardwareOf = (c: Compute): { text: string; code: boolean } => {
  const s = c.spec.specs[0]
  const instance = boundOf(c)?.instance
  if (instance && instance.replace(/[\s-]/g, '').toLowerCase() !== machineOf(c).replace(/[\s-]/g, '').toLowerCase()) return { text: instance, code: true }
  const cpus = c.offer?.cpus ?? s?.cpus
  const memory = c.offer?.memory_gb ?? s?.memory_gb
  return { text: [cpus ? `${cpus} vCPU` : null, memory ? `${memory} GB` : null].filter(Boolean).join(', '), code: false }
}

/** The floor and the ceiling a compute was held to, where they were not simply the size it opened at. */
const boundsOf = (c: Compute): string => {
  const b = c.spec.nodes
  const floor = b.min ?? b.initial
  return b.max ? `elastic ${floor} to ${b.max}` : floor !== b.initial ? `floor ${floor}` : ''
}

/** What went wrong with the calls, worst first — nothing where nothing did. */
const outcomeOf = (c: Compute): string =>
  [failedOf(c) ? `${failedOf(c)} failed` : null, c.tasks.cancelled ? `${c.tasks.cancelled} cancelled` : null, c.tasks.indeterminate ? `${c.tasks.indeterminate} unknown` : null]
    .filter(Boolean)
    .join(', ')

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

/** The filters worth a menu: three questions asked rarely, instead of three controls above two rows. */
function Filters({ providers, accels }: { providers: readonly string[]; accels: readonly string[] }) {
  const f = useStore((s) => s.hist)
  const setUi = useStore((s) => s.setUi)
  const [open, setOpen] = useState(false)
  const box = useRef<HTMLDivElement>(null)
  const set = (patch: Partial<HistoryFilters>) => setUi({ hist: { ...f, ...patch } })
  const on = f.cause !== 'all' || f.provider !== 'all' || f.accel !== 'all'

  useEffect(() => {
    if (!open) return
    const away = (e: MouseEvent) => {
      if (!box.current?.contains(e.target as Node)) setOpen(false)
    }
    const escape = (e: KeyboardEvent) => e.key === 'Escape' && setOpen(false)
    document.addEventListener('mousedown', away)
    document.addEventListener('keydown', escape)
    return () => {
      document.removeEventListener('mousedown', away)
      document.removeEventListener('keydown', escape)
    }
  }, [open])

  return (
    <div className="acts" ref={box} style={{ marginLeft: 0 }}>
      <button className="btn sm" aria-expanded={open} onClick={() => setOpen(!open)}>
        Filters{on ? ' ·' : ''}
      </button>
      {open ? (
        <div className="menu" style={{ display: 'grid', gap: 8, padding: 12, minWidth: 220 }}>
          <label className="field">
            <span className="cap">Ended because</span>
            <select className="search" value={f.cause} onChange={(e) => set({ cause: CAUSES.find(([v]) => v === e.target.value)?.[0] ?? 'all' })}>
              {CAUSES.map(([v, label]) => (
                <option key={v} value={v}>
                  {label}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span className="cap">Provider</span>
            <select className="search" value={f.provider} onChange={(e) => set({ provider: e.target.value })}>
              {['all', ...providers].map((v) => (
                <option key={v} value={v}>
                  {v === 'all' ? 'any provider' : v}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span className="cap">Accelerator</span>
            <select className="search" value={f.accel} onChange={(e) => set({ accel: e.target.value })}>
              {['all', ...accels].map((v) => (
                <option key={v} value={v}>
                  {v === 'all' ? 'any accelerator' : v.toUpperCase()}
                </option>
              ))}
            </select>
          </label>
        </div>
      ) : null}
    </div>
  )
}

/** What has ended: a row per compute, why it went, and what it cost. A row opens it. */
export function History() {
  const navigate = useNavigate()
  const past = useStore((s) => s.history)
  const pages = useHistory()
  const pageHistory = useStore((s) => s.pageHistory)
  const f = useStore((s) => s.hist)
  const setUi = useStore((s) => s.setUi)
  const list = histList(past, f)
  const providers = [...new Set(past.map(providerOf))].sort()
  const accels = [...new Set(past.map(accelOf))].sort()

  return (
    <section className="card">
      <div className="chead" style={{ marginBottom: 4 }}>
        <span className="h">Ended</span>
        <span className="sub">{pages?.total ?? past.length}</span>
        <div className="row wrap spread" style={{ gap: 8 }}>
          <input className="search" placeholder="name or id" value={f.q} autoComplete="off" style={{ minWidth: 150 }} onChange={(e) => setUi({ hist: { ...f, q: e.target.value } })} />
          <Filters providers={providers} accels={accels} />
          <div className="pick">
            {(Object.keys(SINCE) as HistoryFilters['since'][]).map((v) => (
              <button key={v} aria-selected={f.since === v} onClick={() => setUi({ hist: { ...f, since: v } })}>
                {v === 'all' ? 'all time' : v}
              </button>
            ))}
          </div>
        </div>
      </div>
      <TableScroll label="Ended">
        <table>
          <thead>
            <tr>
              <th>Compute</th>
              <th>Provider</th>
              <th>Specs</th>
              <th className="right">Nodes</th>
              <th className="right">Calls</th>
              <th className="right">Cost</th>
              <th className="right">Ended</th>
            </tr>
          </thead>
          <tbody>
            {list.length ? (
              list.map((c) => {
                const ran = ranOf(c)
                const bound = boundOf(c)
                const under = hardwareOf(c)
                const calls = callsOf(c)
                const cost = c.ended?.cost ?? 0
                return (
                  <tr key={c.id} data-open="" data-tip={c.ended ? dateOf(endedAt(c)) : undefined} onClick={() => navigate(`/computes/${c.id}`)}>
                    <td>
                      <b style={{ fontWeight: 600 }}>{c.name ?? c.id}</b>
                      <span className="sub mono">{c.id}</span>
                    </td>
                    <td className="nowrap">
                      {bound?.kind ?? '—'}
                      <span className="sub">{bound?.region ?? 'any region'}</span>
                    </td>
                    <td className="nowrap">
                      {machineOf(c)}
                      <span className={under.code ? 'sub mono' : 'sub'}>{under.text}</span>
                    </td>
                    <td className="right nowrap">
                      {targetOf(c)}
                      <span className="sub">{boundsOf(c)}</span>
                    </td>
                    <td className="right nowrap">
                      {calls || '—'}
                      <span className="sub" style={{ color: failedOf(c) ? 'var(--bad)' : undefined }}>{outcomeOf(c)}</span>
                    </td>
                    <td className="right nowrap">
                      {cost ? money(cost) : '—'}
                      <span className="sub">
                        {cost && ran > 60e3 ? `${money(cost / (ran / HOUR), cost / (ran / HOUR) < 10 ? 2 : 0)}/h · ` : ''}
                        {c.spec.allocation.replace(/_/g, ' ')}
                      </span>
                    </td>
                    <td className="right nowrap">
                      {c.ended ? ago(endedAt(c)) : '—'}
                      {ran < 60e3 ? null : <span className="sub">ran {dur(ran)}</span>}
                    </td>
                  </tr>
                )
              })
            ) : (
              <tr>
                <td colSpan={7} className="sub" style={{ padding: '22px 0', textAlign: 'center' }}>
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
