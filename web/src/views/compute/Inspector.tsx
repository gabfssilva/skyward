import { useNavigate, useParams } from 'react-router-dom'
import { useStore, computeById, isLive } from '../../state/store'
import { readyOf } from '../../state/model'
import { Bars } from '../../ui/charts'
import type { BarItem } from '../../ui/charts'

const NONE: never[] = []

export function Inspector() {
  const { id = '' } = useParams()
  const navigate = useNavigate()
  const c = useStore((s) => computeById(s, id))
  const live = useStore((s) => isLive(s, id))
  const nodes = useStore((s) => s.nodes[id]) ?? NONE
  const metrics = useStore((s) => s.metrics)
  const computes = useStore((s) => s.computes)
  const history = useStore((s) => s.history)
  const sel = useStore((s) => s.sel)
  const choose = useStore((s) => s.pick)
  if (!c) return null

  const ready = readyOf(nodes)
  const items = (metric: 'gpu' | 'temp', n: number, asc: boolean): BarItem[] =>
    ready
      .map((x) => ({ rank: x.rank, value: metrics[`${id}/${x.rank}`]?.[metric] ?? 0 }))
      .sort((a, b) => (asc ? a.value - b.value : b.value - a.value))
      .slice(0, n)
  const pick = (rank: number) => choose(sel && sel.rank === rank && sel.computeId === id ? null : { computeId: id, rank })

  const name = c.name ?? c.id
  const kin = [...computes.map((k) => ({ k, live: true })), ...history.map((k) => ({ k, live: false }))].filter(({ k }) => k.name === c.name && k.id !== c.id)

  return (
    <>
      {live ? (
        <section className="card tight">
          <div className="cap">Stragglers</div>
          <div style={{ marginTop: 8 }}>
            <Bars items={items('gpu', 5, true)} metric="gpu" onPick={pick} />
          </div>
          <div className="divider" />
          <div className="cap">Hottest cards</div>
          <div style={{ marginTop: 8 }}>
            <Bars items={items('temp', 3, false)} metric="temp" onPick={pick} />
          </div>
        </section>
      ) : null}
      <section className="card tight">
        <div className="cap">Also named {name}</div>
        <div style={{ marginTop: 8 }}>
          {kin.length ? (
            kin.map(({ k, live: running }) => (
              <div className="kv" key={k.id}>
                <button className="row" style={{ gap: 6 }} onClick={() => navigate(`/computes/${k.id}`)}>
                  <i className={`dot ${k.status.state}`} />
                  <span className="mono">{k.id}</span>
                </button>
                <span className="mono faint">{running ? 'running now' : k.status.state}</span>
              </div>
            ))
          ) : (
            <div className="sub">Nothing else carried this name.</div>
          )}
        </div>
      </section>
    </>
  )
}
