import type { ReactNode } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { api } from '../../api/client'
import type { Node, Task } from '../../api/client'
import { accrued, ago, clock, dur, gpusOf, median, money, ms, nodeLive, rateOf, readyOf, targetOf, taskOf, execsOf } from '../../state/model'
import type { ExecRow } from '../../state/model'
import { bandOf, computeById, nodesOf, useStore, valuesOf } from '../../state/store'
import { Band } from '../../ui/charts'
import { Comb } from '../../ui/comb'
import { combNodes } from '../../state/nodes'
import { Icon } from '../../ui/icons'
import { Fn, Pill } from '../../ui/primitives'
import { Stage as TasksStage } from './Stage'

const attemptOf = (t: Task): number => t.executions.reduce((n, e) => Math.max(n, e.ordinal), 1)

const barFill = (e: ExecRow): string | undefined =>
  e.state === 'failed' ? 'var(--bad)' : e.state === 'started' ? 'var(--boot)' : undefined

function useTask(id: string | undefined) {
  const tasks = useStore((s) => s.tasks)
  return id ? taskOf(tasks, id) : null
}

export function TaskStage() {
  const { id } = useParams()
  const navigate = useNavigate()
  const found = useTask(id)
  const state = useStore((s) => s)
  const sel = useStore((s) => s.sel)
  const pick = useStore((s) => s.pick)
  if (!found) return <TasksStage />
  const { t, computeId } = found
  const c = computeById(state, computeId)
  if (!c) return <TasksStage />
  const nodes = nodesOf(state, computeId)
  const ex = execsOf(t, nodes)
  const ranks = new Set(ex.map((e) => e.rank))
  const top = Math.max(...ex.map((e) => e.ms), 1)
  const failed = ex.filter((e) => e.state === 'failed')
  const sorted = ex.slice().sort((a, b) => b.ms - a.ms)
  const slowest = sorted[0]
  const fastest = sorted[sorted.length - 1]
  return (
    <>
      <section className="card">
        <div className="combhead">
          <button className="btn sm ghost" onClick={() => navigate('/tasks')}>
            <Icon name="back" />
            Tasks
          </button>
          <Fn sha={t.function} size={18} weight={700} />
          <Pill state={t.state} />
          <span className="sub">{t.dispatch === 'all' ? 'on every node of' : t.dispatch === 'stream' ? 'streaming from' : 'on one node of'}</span>
          <button className="row" style={{ gap: 6, fontWeight: 600 }} onClick={() => navigate(`/computes/${c.id}`)}>
            <i className={`dot ${c.status.state}`} />
            {c.name}
          </button>
          <span className="mono faint" style={{ marginLeft: 'auto' }}>
            {t.id} · attempt {attemptOf(t)} · {ago(ms(t.submitted_at))}
          </span>
        </div>
        <Comb
          nodes={combNodes(computeId, nodes, state.metrics, state.progress)}
          computeId={computeId}
          name={c.name ?? undefined}
          only={ranks}
          selected={sel && sel.computeId === computeId ? sel.rank : null}
          onPick={(rank) => pick({ computeId, rank })}
        />
        <div style={{ marginTop: 14 }} className="legend">
          <span>
            <i style={{ background: 'var(--ok)' }} />
            <b>{ranks.size}</b> ranks ran it
          </span>
          {failed.length ? (
            <span>
              <i style={{ background: 'var(--bad)' }} />
              <b>{failed.length}</b> failed, retried
            </span>
          ) : null}
          <span>
            <i style={{ background: 'var(--sunk)' }} />
            not placed
          </span>
        </div>
      </section>

      <section className="card">
        <div className="combhead">
          <b>Time per rank</b>
          <span className="sub">{t.finished_at && slowest && fastest ? `slowest ${dur(slowest.ms)} · fastest ${dur(fastest.ms)}` : 'still running'}</span>
        </div>
        <div className="bars">
          {sorted.slice(0, 12).map((e) => (
            <div
              key={`${e.rank}/${e.ordinal}`}
              className="barrow"
              style={{ cursor: 'pointer', gridTemplateColumns: '64px 1fr 64px' }}
              onClick={() => pick({ computeId, rank: e.rank })}
            >
              <span className="mono faint">
                rank {e.rank}
                {e.ordinal > 1 ? ` #${e.ordinal}` : ''}
              </span>
              <span className="track">
                <i style={{ width: `${(e.ms / top) * 100}%`, background: barFill(e) }} />
              </span>
              <span className="mono right">{dur(e.ms)}</span>
            </div>
          ))}
        </div>
        {sorted.length > 12 ? (
          <div className="sub" style={{ marginTop: 8 }}>
            {sorted.length - 12} more ranks within {dur(sorted[12]!.ms)}
          </div>
        ) : null}
      </section>

      {failed.length ? (
        <section className="card">
          <div className="combhead">
            <b>Failures</b>
          </div>
          {failed.map((e) => (
            <div key={`${e.rank}/${e.ordinal}`} className="strip bad" style={{ marginBottom: 8, alignItems: 'flex-start' }}>
              <Icon name="alert" />
              <div>
                <b style={{ fontWeight: 700 }}>
                  rank {e.rank} · attempt {e.ordinal} · {dur(e.ms)}
                </b>
                <div className="mono" style={{ marginTop: 3 }}>
                  {e.error}
                </div>
                {t.executions.find((r) => r.rank === e.rank && r.ordinal > e.ordinal) ? (
                  <div className="sub" style={{ marginTop: 4 }}>
                    retried on the same rank, attempt {e.ordinal + 1} succeeded
                  </div>
                ) : null}
              </div>
            </div>
          ))}
        </section>
      ) : null}
    </>
  )
}

export function TaskInspector() {
  const { id } = useParams()
  const navigate = useNavigate()
  const found = useTask(id)
  const state = useStore((s) => s)
  const setUi = useStore((s) => s.setUi)
  const reloadCompute = useStore((s) => s.reloadCompute)
  if (!found) return null
  const { t, computeId } = found
  const c = computeById(state, computeId)
  if (!c) return null
  const nodes = nodesOf(state, computeId)
  const ex = execsOf(t, nodes)
  const worker = c.spec.worker
  const cancel = async () => {
    await api.cancelTask(t.id)
    await reloadCompute(computeId)
  }
  const again = async () => {
    await api.retry(t.id)
    await reloadCompute(computeId)
  }
  return (
    <>
      <section className="card tight">
        <div className="cap">Task</div>
        <div style={{ marginTop: 6 }}>
          <div className="kv">
            <span className="faint">function</span>
            <Fn sha={t.function} weight={400} />
          </div>
          <div className="kv">
            <span className="faint">compute</span>
            <span className="mono">{c.name}</span>
          </div>
          <div className="kv">
            <span className="faint">dispatch</span>
            <span className="mono">{t.dispatch}</span>
          </div>
          <div className="kv">
            <span className="faint">ranks</span>
            <span className="mono">{new Set(ex.map((e) => e.rank)).size}</span>
          </div>
          <div className="kv">
            <span className="faint">executions</span>
            <span className="mono">{ex.length}</span>
          </div>
          <div className="kv">
            <span className="faint">submitted</span>
            <span className="mono">{clock(ms(t.submitted_at))}</span>
          </div>
          <div className="kv">
            <span className="faint">{t.finished_at ? 'finished' : 'running for'}</span>
            <span className="mono">{t.finished_at ? clock(ms(t.finished_at)) : dur(Date.now() - ms(t.submitted_at))}</span>
          </div>
          <div className="kv">
            <span className="faint">executor</span>
            <span className="mono">
              {worker?.executor ?? 'thread'} × {worker?.concurrency ?? 1}
            </span>
          </div>
        </div>
        <div className="row" style={{ gap: 6, marginTop: 12 }}>
          {t.state === 'running' ? (
            <button className="btn sm danger" onClick={() => void cancel()}>
              <Icon name="close" />
              Cancel
            </button>
          ) : (
            <button className="btn sm" onClick={() => void again()}>
              <Icon name="refresh" />
              Run again
            </button>
          )}
          <button
            className="btn sm"
            onClick={() => {
              setUi({ dock: 'logs', dockMin: false })
              navigate(`/computes/${computeId}`)
            }}
          >
            <Icon name="logs" />
            Logs
          </button>
        </div>
      </section>
      <section className="card tight">
        <div className="cap">Cluster GPU while it ran</div>
        <div style={{ marginTop: 8 }}>
          <Band hist={bandOf(state, computeId)} />
        </div>
      </section>
    </>
  )
}

/** The rail of a task: its compute, exactly as `renderRail` draws a compute. */
export function TaskRail() {
  const { id } = useParams()
  const navigate = useNavigate()
  const found = useTask(id)
  const state = useStore((s) => s)
  const openSheet = useStore((s) => s.openSheet)
  const setUi = useStore((s) => s.setUi)
  if (!found) return null
  const c = computeById(state, found.computeId)
  if (!c) return null
  const nodes: Node[] = nodesOf(state, c.id)
  const v = valuesOf(state, c.id, 'gpu')
  const meters: readonly (readonly [ReactNode, string])[] = [
    [
      <>
        {money(rateOf(nodes), 2)}
        <small>/h</small>
      </>,
      'per hour',
    ],
    [
      <>
        {readyOf(nodes).length}
        <small>/{targetOf(c)}</small>
      </>,
      'nodes ready',
    ],
    [
      <>
        {Math.round(median(v) || 0)}
        <small>%</small>
      </>,
      'median GPU',
    ],
    [<>{gpusOf(nodes)}</>, 'GPUs attached'],
    [<>{money(accrued(c, nodes), 0)}</>, `spent in ${dur(Date.now() - ms(c.created_at))}`],
  ]
  return (
    <>
      <div className="gauge-r">
        <b style={{ fontSize: 28 }}>{c.name}</b>
        <span>
          <Pill state={c.status.state} /> · gen {c.generation} · {c.id}
        </span>
      </div>
      {meters.map(([value, label]) => (
        <div className="gauge-r" key={label}>
          <b>{value}</b>
          <span>{label}</span>
        </div>
      ))}
      <div className="spacer row" style={{ gap: 6, position: 'relative' }}>
        <button className="btn" onClick={() => openSheet({ kind: 'scale', computeId: c.id })}>
          <Icon name="scale" />
          Scale
        </button>
        <button className="btn" onClick={() => setUi({ dock: 'shell', dockMin: false })}>
          <Icon name="shell" />
          Shell
        </button>
        <button
          className="btn danger"
          onClick={() =>
            openSheet({
              kind: 'confirm',
              title: `Delete ${c.name}?`,
              body: `${nodes.filter(nodeLive).length} machines are terminated at the provider.`,
              confirm: 'Delete',
              danger: true,
              onConfirm: () => {
                void api.deleteCompute(c.id).then(() => navigate('/computes'))
              },
            })
          }
        >
          <Icon name="trash" />
          Delete
        </button>
      </div>
    </>
  )
}
