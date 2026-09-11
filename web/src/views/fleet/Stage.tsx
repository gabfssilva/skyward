import { useNavigate } from 'react-router-dom'
import { useStore } from '../../state/store'
import { dur, median, money, ms, rateOf, specLine } from '../../state/model'
import { Icon } from '../../ui/icons'
import { Legend } from '../../ui/primitives'
import { LiveComb } from './LiveComb'
import { valuesFor } from '../../state/nodes'

export function Stage() {
  const navigate = useNavigate()
  const computes = useStore((s) => s.computes)
  const nodesByCompute = useStore((s) => s.nodes)
  const metrics = useStore((s) => s.metrics)
  const openSheet = useStore((s) => s.openSheet)

  if (!computes.length)
    return (
      <section className="card">
        <div className="empty">
          <Icon name="fleet" />
          <b>Nothing is running</b>
          <span>A compute is a set of machines the daemon keeps for you. Buy the first one from the market.</span>
          <button className="btn primary" style={{ marginTop: 8 }} onClick={() => openSheet({ kind: 'wizard' })}>
            <Icon name="plus" />
            New compute
          </button>
        </div>
      </section>
    )

  return (
    <>
      {computes.map((c) => {
        const nodes = nodesByCompute[c.id] ?? []
        return (
          <section className="card" key={c.id}>
            <div className="combhead">
              <button className="row" style={{ gap: 8 }} onClick={() => navigate(`/computes/${c.id}`)}>
                <i className={`dot ${c.status.state}`} />
                <b>{c.name}</b>
              </button>
              <span className="sub">
                {nodes.length} {nodes.length === 1 ? 'node' : 'nodes'} · {specLine(c, nodes)}
              </span>
              <span className="mono faint" style={{ marginLeft: 'auto' }}>
                {money(rateOf(nodes))}/h · median GPU {Math.round(median(valuesFor(c.id, nodes, metrics, 'gpu')) || 0)}% · {dur(Date.now() - ms(c.created_at))}
              </span>
            </div>
            <LiveComb compute={c} nodes={nodes} />
            <div style={{ marginTop: 12 }}>
              <Legend states={nodes.map((n) => n.state)} />
            </div>
          </section>
        )
      })}
    </>
  )
}
