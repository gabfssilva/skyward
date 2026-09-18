import { useNavigate } from 'react-router-dom'
import type { Compute } from '../../api/client'
import { useStore } from '../../state/store'
import { acceleratedOf, boundOf, loadOf, machineOf, median, money, rateOf } from '../../state/model'
import { nodeCells } from '../../state/nodes'
import { Hive, scaleFor } from '../../ui/comb'
import { PageHead } from '../../ui/head'
import { Empty, Pill, useMeasure } from '../../ui/primitives'
import { openWizard } from '../../sheets'
import { Attention } from './Attention'
import { History } from './History'

/** Where the offers say the machines come from: one kind, or all of the kinds the specs allow. */
const providersOf = (c: Compute): string => [...new Set(c.spec.specs.map((s) => s.provider.kind))].join(', ') || '—'

/**
 * The home, in the order the questions come: what is it costing me, what is wrong, what is running, what has ended.
 *
 * Every compute is one tile, and every tile draws its hive at the same cell size — so 284 nodes take up more of
 * the page than 64, and a compute of one node is one small hexagon. A drawing that scaled each compute to fit
 * its own box made the smallest compute the largest thing on the screen.
 */
export function Stage() {
  const navigate = useNavigate()
  const computes = useStore((s) => s.computes)
  const nodesByCompute = useStore((s) => s.nodes)
  const readings = useStore((s) => s.readings)
  const progress = useStore((s) => s.progress)
  const costs = useStore((s) => s.costs)
  /* the daemon's count keeps the card and its filters up when a cause narrows the first page to nothing */
  const ended = useStore((s) => s.history.length > 0 || !!s.histPages?.total)
  const [box, { w, h }] = useMeasure<HTMLSpanElement>()

  const ready = computes.flatMap((c) => (nodesByCompute[c.id] ?? []).filter((n) => n.state === 'ready').map((n) => ({ c, n })))
  const busy = ready.map(({ c, n }) => loadOf(readings[`${c.id}/${n.rank}`], acceleratedOf(c)))
  const loads = busy.filter((v): v is number => v !== null)
  const idle = ready.filter((_, i) => (busy[i] ?? 100) < 25)
  const rate = computes.reduce((sum, c) => sum + rateOf(nodesByCompute[c.id] ?? []), 0)
  const spent = computes.reduce((sum, c) => sum + (costs[c.id] ?? 0), 0)
  const cards = computes.reduce((sum, c) => sum + (acceleratedOf(c) ? (nodesByCompute[c.id] ?? []).filter((n) => n.state === 'ready').length * (boundOf(c)?.accelerator_count ?? 1) : 0), 0)
  /* a fleet of one kind of machine says which gauge it is being read on; one of both says neither word */
  const gauges = new Set(ready.map(({ c }) => (acceleratedOf(c) ? 'accelerator' : 'CPU')))
  const gauge = gauges.size === 1 ? [...gauges][0]! : 'load'

  const tiles = computes
    .map((c) => ({ c, nodes: nodesByCompute[c.id] ?? [], rate: rateOf(nodesByCompute[c.id] ?? []) }))
    .sort((a, b) => b.rate - a.rate)
  const size = w > 0 ? scaleFor(tiles.map((t) => Math.max(1, t.nodes.length)), w, h) : 0

  return (
    <>
      <PageHead primary={{ label: 'New compute', icon: 'plus', onClick: () => openWizard() }}>
        <div className="sum">
          <span className="hero">
            {money(rate, 2)}
            <small>/h</small>
          </span>
          {spent ? (
            <span>
              <b>{money(spent, spent < 10 ? 2 : 0)}</b> spent
            </span>
          ) : null}
          <span>
            <b>{ready.length}</b> node{ready.length === 1 ? '' : 's'}
          </span>
          {cards ? (
            <span>
              <b>{cards}</b> accelerator{cards === 1 ? '' : 's'}
            </span>
          ) : null}
          {loads.length ? (
            <span>
              <b>{Math.round(median(loads))}%</b> median {gauge}
            </span>
          ) : null}
          {idle.length ? (
            <span>
              <b>{idle.length} idle</b>, {money(idle.reduce((sum, { n }) => sum + (n.price_per_hour ?? 0), 0))}/h
            </span>
          ) : null}
        </div>
      </PageHead>

      <Attention />

      {computes.length ? (
        <>
          <div className="sec">
            <span className="h">Running</span>
            <span className="sub">{computes.length}</span>
          </div>
          <div className="tiles">
            {tiles.map(({ c, nodes, rate: hourly }, i) => (
              <button key={c.id} className="card tile" onClick={() => navigate(`/computes/${c.id}`)}>
                <span className="hv" ref={i === 0 ? box : undefined}>
                  {size > 0 && nodes.length ? (
                    <Hive cells={nodeCells(c, nodes, readings, progress)} computeId={c.id} size={size} label={`${nodes.length} nodes of ${c.name ?? c.id}`} />
                  ) : null}
                </span>
                <span className="row">
                  <span className="name">{c.name ?? c.id}</span>
                  <span className="spread">
                    <Pill state={c.status.state} />
                  </span>
                </span>
                <span className="shape">
                  {nodes.length || c.spec.nodes.initial} × {machineOf(c)} on {providersOf(c)}
                </span>
                <span className="tf">
                  <span>
                    <b>{money(hourly, 2)}</b>/h
                  </span>
                  {costs[c.id] ? <span>{money(costs[c.id]!, costs[c.id]! < 10 ? 2 : 0)} spent</span> : null}
                  <span>{c.tasks.running ? `${c.tasks.running} running` : c.tasks.queued ? `${c.tasks.queued} queued` : 'idle'}</span>
                </span>
              </button>
            ))}
          </div>
        </>
      ) : (
        <section className="card">
          <Empty icon="fleet" title="Nothing is running">
            A compute is a set of machines the daemon keeps for you. Buy the first one from the market.
          </Empty>
        </section>
      )}

      {ended ? <History /> : null}
    </>
  )
}
