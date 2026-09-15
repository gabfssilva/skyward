import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { api } from '../../api/client'
import type { Compute, Task } from '../../api/client'
import { ago, clock, dur, ms, taskOf, execsOf } from '../../state/model'
import type { ExecRow } from '../../state/model'
import { bandOf, computeById, isLive, nodesOf, useStore } from '../../state/store'
import { Band } from '../../ui/charts'
import { Comb } from '../../ui/comb'
import { combNodes } from '../../state/nodes'
import { Icon } from '../../ui/icons'
import { Fn, Pill } from '../../ui/primitives'
import { ComputeActions, ComputeStats } from '../compute/Rail'
import { Stage as TasksStage } from './Stage'

const attemptOf = (t: Task): number => t.executions.reduce((n, e) => Math.max(n, e.ordinal), 1)

const barFill = (e: ExecRow): string | undefined =>
  e.state === 'failed' ? 'var(--bad)' : e.state === 'started' ? 'var(--boot)' : undefined

/** What the page is about, once both halves of it are in the store. */
type Subject = { t: Task; computeId: string; c: Compute }

const asking = new Map<string, Promise<void>>()

/** what the daemon does not know: a task it never had, or a compute it no longer keeps */
const missing = new Set<string>()

/** Ask for one id once, however many of the page's parts want it. */
const once = (id: string, work: () => Promise<void>): Promise<void> => {
  const running = asking.get(id) ?? work().finally(() => asking.delete(id))
  asking.set(id, running)
  return running
}

/** File a task the page fetched where the store keeps tasks, its compute's list newest first. */
function file(t: Task): void {
  const { tasks, setEntities } = useStore.getState()
  const known = tasks[t.compute_id] ?? []
  if (known.some((x) => x.id === t.id)) return
  setEntities({ tasks: { ...tasks, [t.compute_id]: [t, ...known].sort((a, b) => ms(b.submitted_at) - ms(a.submitted_at)) } })
}

/**
 * The task the page is about, and the compute it ran on.
 *
 * History is paged, so a link can name a task no page carried, and a task can name a
 * compute nothing loaded. Each is asked for once — the task from the daemon, the compute
 * through ``learn`` — and the compute is learnt before the task is filed, so the page is
 * never handed a task whose compute is still on its way. ``looking`` says the daemon may
 * yet answer, which is what tells a page filling in from one that never will.
 */
function useTask(id: string | undefined): { found: Subject | null; looking: boolean } {
  const tasks = useStore((s) => s.tasks)
  const found = id ? taskOf(tasks, id) : null
  const computeId = found?.computeId
  const c = useStore((s) => (computeId ? computeById(s, computeId) : undefined))
  const [, bump] = useState(0)
  const have = found !== null
  const named = c !== undefined

  useEffect(() => {
    if (!id || have || missing.has(id)) return
    void once(id, async () => {
      const t = await api.task(id).catch(() => null)
      if (!t) return void missing.add(id)
      await useStore.getState().learn([t.compute_id])
      file(t)
    }).then(() => bump((n) => n + 1))
  }, [id, have])

  useEffect(() => {
    if (!computeId || named || missing.has(computeId)) return
    void once(computeId, async () => {
      await useStore.getState().learn([computeId])
      if (!computeById(useStore.getState(), computeId)) missing.add(computeId)
    }).then(() => bump((n) => n + 1))
  }, [computeId, named])

  const lost = !!id && (missing.has(id) || (computeId !== undefined && missing.has(computeId)))
  return { found: found && c ? { ...found, c } : null, looking: !lost }
}

export function TaskStage() {
  const { id } = useParams()
  const navigate = useNavigate()
  const { found, looking } = useTask(id)
  const state = useStore((s) => s)
  if (!found) return looking ? null : <TasksStage />
  const { t, computeId, c } = found
  const nodes = nodesOf(state, computeId)
  const ex = execsOf(t, nodes)
  const ranks = new Set(ex.map((e) => e.rank))
  const top = Math.max(...ex.map((e) => e.ms), 1)
  const failed = ex.filter((e) => e.state === 'failed')
  const sorted = ex.slice().sort((a, b) => b.ms - a.ms)
  const slowest = sorted[0]
  const fastest = sorted[sorted.length - 1]
  const open = (rank: number) => navigate(`/computes/${computeId}/nodes/${rank}`)
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
          onPick={open}
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
              onClick={() => open(e.rank)}
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
  const { found } = useTask(id)
  const state = useStore((s) => s)
  const setUi = useStore((s) => s.setUi)
  const reloadCompute = useStore((s) => s.reloadCompute)
  if (!found) return null
  const { t, computeId, c } = found
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
              const ranks = new Set(ex.map((e) => e.rank))
              const rank = ranks.size === 1 ? [...ranks][0]! : 'all'
              setUi({ act: { ...state.act, kind: 'logs', compute: computeId, rank } })
              navigate('/activity')
            }}
          >
            <Icon name="logs" />
            Logs in Activity
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

/** The task page's compact rail: the crumb down to the function, then the compute's stats and actions, as on a node. */
export function TaskRail() {
  const { id } = useParams()
  const navigate = useNavigate()
  const { found } = useTask(id)
  const state = useStore((s) => s)
  if (!found) return null
  const { t, c } = found
  const nodes = nodesOf(state, c.id)
  const live = isLive(state, c.id)
  return (
    <>
      <div className="crumb">
        <button aria-label="Computes" onClick={() => navigate('/')}>
          <Icon name="fleet" />
        </button>
        <span className="sep">/</span>
        <button onClick={() => navigate(`/computes/${c.id}`)}>
          <b>{c.name ?? c.id}</b>
        </button>
        <Pill state={c.status.state} />
        <span className="sep">/</span>
        <Fn sha={t.function} />
        <Pill state={t.state} />
      </div>
      <ComputeStats c={c} nodes={nodes} live={live} />
      <ComputeActions c={c} nodes={nodes} live={live} />
    </>
  )
}
