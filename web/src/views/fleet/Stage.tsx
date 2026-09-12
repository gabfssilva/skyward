import { useNavigate } from 'react-router-dom'
import { useStore } from '../../state/store'
import { rateOf } from '../../state/model'
import { combNodes } from '../../state/nodes'
import { Icon } from '../../ui/icons'
import { Hives, type HiveItem } from '../../ui/comb'
import { History } from './History'

/** The home: every live compute as one hive, then what has ended. */
export function Stage() {
  const navigate = useNavigate()
  const computes = useStore((s) => s.computes)
  const nodesByCompute = useStore((s) => s.nodes)
  const tasks = useStore((s) => s.tasks)
  const metrics = useStore((s) => s.metrics)
  const progress = useStore((s) => s.progress)
  const openSheet = useStore((s) => s.openSheet)
  const pick = useStore((s) => s.pick)
  /* the daemon's count keeps the card and its filters up when a cause narrows the first page to nothing */
  const ended = useStore((s) => s.history.length > 0 || !!s.histPages?.total)

  if (!computes.length)
    return (
      <>
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
        {ended ? <History /> : null}
      </>
    )

  const items: HiveItem[] = computes.map((c) => {
    const nodes = nodesByCompute[c.id] ?? []
    const busy: Record<number, number> = {}
    for (const t of tasks[c.id] ?? [])
      if (t.state === 'running') for (const e of t.executions) if (e.state === 'started') busy[e.rank] = (busy[e.rank] ?? 0) + 1
    return {
      id: c.id,
      name: c.name ?? c.id,
      state: c.status.state,
      rate: rateOf(nodes),
      slots: Math.max(1, c.spec.worker?.concurrency ?? 1),
      nodes: combNodes(c.id, nodes, metrics, progress),
      busy,
    }
  })

  return (
    <>
      <section className="card">
        <Hives
          items={items}
          onPick={(computeId, rank) => {
            pick({ computeId, rank })
            navigate(`/computes/${computeId}`)
          }}
          onOpen={(computeId) => navigate(`/computes/${computeId}`)}
        />
      </section>
      {ended ? <History /> : null}
    </>
  )
}
