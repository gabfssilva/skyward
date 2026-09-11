import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import type { Compute } from '../../api/client'
import { useStore } from '../../state/store'
import { accrued, ago, dur, median, money, ms, rateOf, specLine, targetOf } from '../../state/model'
import { Icon } from '../../ui/icons'
import { LiveComb } from '../fleet'
import { valuesFor } from '../../state/nodes'

const OPEN = 'sky-history-open'

const stored = (): boolean => {
  try {
    return localStorage.getItem(OPEN) === '1'
  } catch {
    return false
  }
}

/** What is left of a compute the daemon has released. */
function Past({ computes }: { computes: readonly Compute[] }) {
  const navigate = useNavigate()
  const [open, setOpen] = useState(stored)
  if (!computes.length) return null

  const toggle = () => {
    const next = !open
    setOpen(next)
    try {
      localStorage.setItem(OPEN, next ? '1' : '0')
    } catch {
      /* a browser that keeps nothing still gets the toggle */
    }
  }

  return (
    <section className="card">
      <button className="btn sm ghost" onClick={toggle}>
        <Icon name={open ? 'hide' : 'show'} />
        History · {computes.length}
      </button>
      {open
        ? computes.map((c) => (
            <div
              key={c.id}
              className="strip"
              style={{
                marginTop: 8,
                background: 'var(--sunk)',
                display: 'grid',
                gridTemplateColumns: 'minmax(0,48ch) minmax(0,1.4fr) 110px 210px',
                gap: 14,
                alignItems: 'center',
                cursor: 'pointer',
              }}
              onClick={() => navigate(`/computes/${c.id}`)}
            >
              <div className="row" style={{ gap: 6 }}>
                <i className="dot deleted" />
                <b className="trunc" style={{ fontWeight: 600, color: 'var(--muted)' }}>
                  {c.name ?? c.id}
                </b>
              </div>
              <div className="sub trunc">{specLine(c, [])}</div>
              <div className="mono faint trunc">{c.spec.specs[0]?.provider.kind ?? '—'}</div>
              <div className="mono faint right" style={{ whiteSpace: 'nowrap' }}>
                {targetOf(c)} nodes · created {ago(ms(c.created_at))}
              </div>
            </div>
          ))
        : null}
    </section>
  )
}

export function Stage() {
  const navigate = useNavigate()
  const computes = useStore((s) => s.computes)
  const nodesByCompute = useStore((s) => s.nodes)
  const metrics = useStore((s) => s.metrics)
  const openSheet = useStore((s) => s.openSheet)
  const history = useStore((s) => s.history)

  return (
    <>
      <section className="card">
        <div className="combhead">
          <b>Computes</b>
          <button className="btn primary" style={{ marginLeft: 'auto' }} onClick={() => openSheet({ kind: 'wizard' })}>
            <Icon name="plus" />
            New compute
          </button>
        </div>
        {computes.map((c) => {
          const nodes = nodesByCompute[c.id] ?? []
          const ready = nodes.filter((n) => n.state === 'ready').length
          return (
            <div
              key={c.id}
              className="strip r2"
              style={{
                alignItems: 'flex-start',
                marginBottom: 8,
                background: 'var(--sunk)',
                display: 'grid',
                gridTemplateColumns: 'minmax(160px,1fr) minmax(0,1.3fr) 210px',
                gap: 14,
              }}
            >
              <div>
                <button className="row" style={{ gap: 6 }} onClick={() => navigate(`/computes/${c.id}`)}>
                  <i className={`dot ${c.status.state}`} />
                  <b style={{ fontWeight: 600 }}>{c.name}</b>
                </button>
                <div className="mono faint" style={{ marginTop: 3 }}>
                  {c.id} · gen {c.generation}
                </div>
                <div className="sub" style={{ marginTop: 5 }}>
                  {specLine(c, nodes)}
                </div>
              </div>
              <div>
                <LiveComb compute={c} nodes={nodes} room={330} />
              </div>
              <div>
                <div className="kv">
                  <span className="faint">nodes</span>
                  <span className="mono">
                    {ready} of {targetOf(c)}
                  </span>
                </div>
                <div className="kv">
                  <span className="faint">median GPU</span>
                  <span className="mono">{Math.round(median(valuesFor(c.id, nodes, metrics, 'gpu')) || 0)}%</span>
                </div>
                <div className="kv">
                  <span className="faint">rate</span>
                  <span className="mono">{money(rateOf(nodes))}/h</span>
                </div>
                <div className="kv">
                  <span className="faint">spent</span>
                  <span className="mono">
                    {money(accrued(c, nodes), 0)} in {dur(Date.now() - ms(c.created_at))}
                  </span>
                </div>
              </div>
            </div>
          )
          })}
      </section>
      <Past computes={history} />
    </>
  )
}
