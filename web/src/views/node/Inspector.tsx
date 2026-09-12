import { useNavigate } from 'react-router-dom'
import { api } from '../../api/client'
import { useStore, historyOf } from '../../state/store'
import { METRICS, UNIT, dur, hiveSize, money, ms } from '../../state/model'
import { combNodes } from '../../state/nodes'
import { Comb } from '../../ui/comb'
import { Spark } from '../../ui/charts'
import { Icon } from '../../ui/icons'
import { Pill } from '../../ui/primitives'
import { VRAM } from './Shell'
import { PhaseProgress, replace, useNode } from './Stage'

/** The node page's inspector: where the node sits in its compute, its facts, its gauges, and what can be done to it. */
export function Inspector() {
  const navigate = useNavigate()
  const found = useNode()
  const state = useStore((s) => s)
  const shell = state.shell
  const setUi = state.setUi
  if (!found) return null
  const { c, n, live, nodes } = found
  const ready = n.state === 'ready'
  const m = state.metrics[`${c.id}/${n.rank}`]
  const accel = n.accelerator ?? c.spec.specs[0]?.accelerator ?? '?'
  const count = c.spec.specs[0]?.accelerator_count ?? 1
  const vram = VRAM[accel]

  const drain = async () => {
    await api.drainNode(c.id, n.id)
    await useStore.getState().reloadCompute(c.id)
    navigate(`/computes/${c.id}`)
  }

  return (
    <>
      <section className="card tight">
        <div className="cap">In {c.name ?? c.id}</div>
        <div style={{ marginTop: 10 }}>
          <Comb
            layout="hive"
            nodes={combNodes(c.id, nodes, state.metrics, state.progress)}
            computeId={c.id}
            name={c.name ?? c.id}
            size={hiveSize(nodes.length, 300, 300)}
            only={new Set([n.rank])}
            onPick={(rank) => navigate(`/computes/${c.id}/nodes/${rank}`)}
          />
        </div>
      </section>
      <section className="card tight">
        <div className="row" style={{ justifyContent: 'space-between' }}>
          <div className="row">
            <span className="mono" style={{ background: 'var(--sunk)', padding: '1px 7px', borderRadius: 5 }}>
              rank {n.rank}
            </span>
            <Pill state={n.state} />
          </div>
        </div>
        <div className="mono faint" style={{ marginTop: 8 }}>
          {n.id}
        </div>
        <div style={{ marginTop: 8 }}>
          <div className="kv">
            <span className="faint">address</span>
            <span className="mono">{n.address ?? '—'}</span>
          </div>
          <div className="kv">
            <span className="faint">machine</span>
            <span className="mono trunc" style={{ maxWidth: 160 }}>
              {n.machine ?? 'not assigned'}
            </span>
          </div>
          <div className="kv">
            <span className="faint">card</span>
            <span className="mono">
              {count}× {accel.toUpperCase()}
              {vram ? ` · ${vram}GB` : ''}
            </span>
          </div>
          <div className="kv">
            <span className="faint">price</span>
            <span className="mono">
              {money(n.price_per_hour ?? 0)}/h {n.market === 'spot' ? 'spot' : 'on demand'}
            </span>
          </div>
          <div className="kv">
            <span className="faint">uptime</span>
            <span className="mono">{ready ? dur(Date.now() - (ms(n.launched_at) || ms(n.created_at))) : '—'}</span>
          </div>
          <div className="kv">
            <span className="faint">generation</span>
            <span className="mono">{n.generation}</span>
          </div>
        </div>
        {ready ? (
          <>
            <div className="divider" />
            <div className="bars" style={{ gap: 8 }}>
              {METRICS.map(([k, l]) => (
                <div className="barrow" key={k} style={{ gridTemplateColumns: '44px 1fr 60px' }}>
                  <span className="faint">{l}</span>
                  <span>
                    <Spark values={historyOf(state, c.id, n.rank, k)} h={26} fmt={(v) => Math.round(v) + UNIT[k]} />
                  </span>
                  <span className="mono right">
                    {Math.round(m?.[k] ?? 0)}
                    {UNIT[k]}
                  </span>
                </div>
              ))}
            </div>
            {live ? (
              <div className="row" style={{ gap: 6, marginTop: 10 }}>
                <button className="btn sm primary" onClick={() => setUi({ shell: true })}>
                  <Icon name="shell" />
                  {shell ? 'Focus the shell' : 'Open a shell'}
                </button>
                <button className="btn sm" onClick={() => void drain()}>
                  Drain
                </button>
              </div>
            ) : null}
          </>
        ) : n.last_error ? (
          <>
            <div className="divider" />
            <div style={{ color: 'var(--bad)', fontSize: 11.5 }}>{n.last_error.message}</div>
            {live ? (
              <button className="btn sm" style={{ marginTop: 9 }} onClick={() => void replace(c, n)}>
                Replace it
              </button>
            ) : null}
          </>
        ) : (
          <>
            <div className="divider" />
            <PhaseProgress nodeId={n.id} />
          </>
        )}
      </section>
    </>
  )
}
