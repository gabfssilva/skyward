import { useParams } from 'react-router-dom'
import { useStore, computeById, isLive } from '../../state/store'
import { valuesFor } from '../../state/nodes'
import { median, ms, readyOf } from '../../state/model'
import { Fn, Pill } from '../../ui/primitives'
import { useNavigate } from 'react-router-dom'
import { Band, Bars } from '../../ui/charts'
import type { BarItem } from '../../ui/charts'
import { Icon } from '../../ui/icons'

const row = (label: string, value: string, tip?: string, wide?: number) => (
  <div className="kv" key={label}>
    <span className="faint">{label}</span>
    <span className={wide ? 'mono trunc' : 'mono'} style={wide ? { maxWidth: wide } : undefined} data-tip={tip}>
      {value}
    </span>
  </div>
)

const NONE: never[] = []
const NO_BAND: never[] = []
const NO_TASKS: never[] = []

export function Inspector() {
  const { id = '' } = useParams()
  const c = useStore((s) => computeById(s, id))
  const live = useStore((s) => isLive(s, id))
  const tasks = useStore((s) => s.tasks[id]) ?? NO_TASKS
  const navigate = useNavigate()
  const nodes = useStore((s) => s.nodes[id]) ?? NONE
  const metrics = useStore((s) => s.metrics)
  const hist = useStore((s) => s.hist[id]) ?? NO_BAND
  const sel = useStore((s) => s.sel)
  const choose = useStore((s) => s.pick)
  const openSheet = useStore((s) => s.openSheet)
  if (!c) return null

  const ready = readyOf(nodes)
  const gpu = valuesFor(id, nodes, metrics, 'gpu')
  const items = (metric: 'gpu' | 'temp', n: number, asc: boolean): BarItem[] =>
    ready
      .map((x) => ({ rank: x.rank, value: metrics[`${id}/${x.rank}`]?.[metric] ?? 0 }))
      .sort((a, b) => (asc ? a.value - b.value : b.value - a.value))
      .slice(0, n)

  const pick = (rank: number) => choose(sel && sel.rank === rank && sel.computeId === id ? null : { computeId: id, rank })

  const spec = c.spec.specs[0]
  const bounds = c.spec.nodes
  const size =
    bounds.initial +
    (bounds.max ? ` (${bounds.min ?? bounds.initial}–${bounds.max})` : bounds.min != null && bounds.min !== bounds.initial ? ` floor ${bounds.min}` : '')
  const base = c.spec.image?.base ?? '—'
  const lease = Math.max(0, Math.round((ms(c.lease.expires_at) - Date.now()) / 1000))

  return (
    <>
      {live ? null : (
        <section className="card tight">
          <div className="cap">Tasks</div>
          <div style={{ marginTop: 6 }}>
            {tasks.length ? (
              tasks.map((t) => (
                <div className="kv" key={t.id} style={{ cursor: 'pointer' }} onClick={() => navigate(`/tasks/${t.id}`)}>
                  <span className="trunc" style={{ maxWidth: 150 }}>
                    <Fn sha={t.function} weight={400} />
                  </span>
                  <Pill state={t.state} />
                </div>
              ))
            ) : (
              <div className="kv">
                <span className="faint">none</span>
              </div>
            )}
          </div>
        </section>
      )}
      {live ? (
      <>
        <section className="card tight">
          <div className="cap">Cluster GPU, last 40 samples</div>
          <div style={{ marginTop: 8 }}>
            <Band hist={hist} />
          </div>
          <div className="row" style={{ justifyContent: 'space-between', marginTop: 6 }}>
            <span className="mono faint">slowest {Math.round(gpu.length ? Math.min(...gpu) : 0)}%</span>
            <span className="mono">median {Math.round(median(gpu) || 0)}%</span>
            <span className="mono faint">fastest {Math.round(gpu.length ? Math.max(...gpu) : 0)}%</span>
          </div>
        </section>
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
      </>
      ) : null}
      <section className="card tight">
        <div className="cap">Spec</div>
        <div style={{ marginTop: 6 }}>
          {row('shape', spec ? `${spec.accelerator_count}× ${(spec.accelerator ?? '?').toUpperCase()} · ${spec.provider.kind}` : '—')}
          {row('size', String(size))}
          {row('allocation', `${c.spec.allocation.replace(/_/g, ' ')} · ${c.spec.selection}`)}
          {row('image', base, base, 170)}
          {row('executor', `${c.spec.worker?.executor ?? 'thread'} × ${c.spec.worker?.concurrency ?? 1}`)}
          {row('plugins', c.spec.plugins.map((p) => p.kind).join(', ') || '—')}
          {live ? row('lease', `renews in ${lease}s`) : null}
          {row('ports', 'none')}
        </div>
        {live ? (
          <button className="btn sm" style={{ marginTop: 10 }} onClick={() => openSheet({ kind: 'ports', computeId: id })}>
            <Icon name="ports" />
            Forward a port
          </button>
        ) : null}
      </section>
    </>
  )
}
