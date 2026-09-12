import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../../api/client'
import type { Offer, Page } from '../../api/client'
import { useStore } from '../../state/store'
import { METRICS, money, offerPerGpu, type MetricKey } from '../../state/model'
import { Histo } from '../../ui/charts'
import { Icon } from '../../ui/icons'
import { Pick } from '../../ui/primitives'
import { valuesFor } from '../../state/nodes'

const HUES = ['--c1', '--c2', '--c3', '--c4', '--c5'] as const

export function Inspector() {
  const navigate = useNavigate()
  const computes = useStore((s) => s.computes)
  const nodesByCompute = useStore((s) => s.nodes)
  const metrics = useStore((s) => s.metrics)
  const costs = useStore((s) => s.costs)
  const metric = useStore((s) => s.metric)
  const setUi = useStore((s) => s.setUi)

  const readyCount = computes.reduce((n, c) => n + (nodesByCompute[c.id] ?? []).filter((x) => x.state === 'ready').length, 0)
  const across = computes.flatMap((c) => valuesFor(c.id, nodesByCompute[c.id] ?? [], metrics, metric))

  const split = computes
    .map((c, i) => ({ c, spent: costs[c.id], hue: HUES[i % 5]! }))
    .sort((a, b) => (b.spent ?? 0) - (a.spent ?? 0))
  const total = split.reduce((sum, x) => sum + (x.spent ?? 0), 0)
  const metered = split.some((x) => x.spent !== undefined)

  const [cheap, setCheap] = useState<Offer[]>([])

  /* ``s.offers`` is the Market view's own slice, so this card asks its own question: the daemon orders by the price of one accelerator and answers in four rows */
  useEffect(() => {
    let live = true
    void api
      .offers({ accelerator: 'h100', sort: 'price', limit: 4 })
      .catch((): Page<Offer> => ({ items: [] }))
      .then((read) => {
        if (live) setCheap(read.items)
      })
    return () => {
      live = false
    }
  }, [])

  return (
    <>
      <section className="card tight">
        <div className="cap">Spent so far</div>
        <div className="gauge-r" style={{ margin: '8px 0 10px' }}>
          <b>{metered ? money(total, total < 10 ? 2 : 0) : '—'}</b>
          <span>since each running compute started</span>
        </div>
        <div style={{ display: 'flex', gap: 3 }}>
          {split.map((x) => (
            <div
              key={x.c.id}
              data-tip={`${x.c.name} · ${x.spent === undefined ? 'not metered yet' : money(x.spent)}`}
              style={{ flex: Math.max(x.spent ?? 0, 0.01), height: 8, borderRadius: 3, background: `var(${x.hue})` }}
            />
          ))}
        </div>
        <div style={{ marginTop: 8 }}>
          {split.map((x) => (
            <div className="kv" key={x.c.id}>
              <button className="row" style={{ gap: 6 }} onClick={() => navigate(`/computes/${x.c.id}`)}>
                <i style={{ width: 9, height: 9, borderRadius: 3, background: `var(${x.hue})` }} />
                {x.c.name}
              </button>
              <span className="mono">{x.spent === undefined ? '—' : money(x.spent)}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="card tight">
        <div className="cap">Across {readyCount} nodes</div>
        <div style={{ marginTop: 8 }}>
          <Pick<MetricKey> value={metric} options={METRICS} onChange={(v) => setUi({ metric: v })} />
        </div>
        <div style={{ marginTop: 9 }}>
          <Histo values={across} metric={metric} />
        </div>
      </section>

      <section className="card tight">
        <div className="row" style={{ justifyContent: 'space-between' }}>
          <span className="cap">Cheapest H100 now</span>
          <button className="btn sm" onClick={() => navigate('/market')}>
            <Icon name="market" />
            Market
          </button>
        </div>
        <div style={{ marginTop: 7 }}>
          {cheap.map((o) => (
            <div className="kv" key={o.id}>
              <span>
                {o.kind}
                <span className="faint mono"> {o.accelerator_count}×</span>
              </span>
              <span className="mono">
                {money(offerPerGpu(o))}
                <span className="faint"> /GPU·h</span>
              </span>
            </div>
          ))}
        </div>
      </section>
    </>
  )
}
