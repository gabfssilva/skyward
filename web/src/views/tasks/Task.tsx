import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { api } from '../../api/client'
import type { Compute, Task } from '../../api/client'
import { acceleratedOf, ago, dur, execsOf, holdersOf, loadGauge, median, ms, runOf, slotsOf, taskOf } from '../../state/model'
import type { ExecRow } from '../../state/model'
import { acrossNodes, axisOf, spanOver, useMetrics } from '../../state/metrics'
import { computeById, nodesOf, useStore } from '../../state/store'
import { Histo, Plot } from '../../ui/charts'
import { Hive, fillRoom } from '../../ui/comb'
import { PageHead } from '../../ui/head'
import { Fn, Pill } from '../../ui/primitives'
import { Stage as TasksStage, ranksOf } from './Stage'

/** How many columns the durations are counted into. */
const BINS = 12

/** The box the lens of ranks fills, whether the task reached one node or three hundred. */
const LENS: readonly [number, number] = [320, 240]

/** How the ranks a task reached are painted: what happened on each, rather than how hard the machine is working. */
const EXEC: Record<string, string> = {
  succeeded: 'color-mix(in oklab, var(--ok) 62%, var(--panel))',
  started: 'var(--boot)',
  created: 'var(--warn)',
  failed: 'var(--bad)',
  retried: 'var(--warn)',
  missing: 'var(--sunk)',
}

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
  const known = tasks[t.compute.id] ?? []
  if (known.some((x) => x.id === t.id)) return
  setEntities({ tasks: { ...tasks, [t.compute.id]: [t, ...known].sort((a, b) => ms(b.submitted_at) - ms(a.submitted_at)) } })
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
      await useStore.getState().learn([t.compute.id])
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

/** What happened on one rank: its last attempt, and whether an earlier one had to be retried. */
const outcome = (rows: readonly ExecRow[]): { state: string; ms: number } => {
  const last = rows[rows.length - 1]
  if (!last) return { state: 'missing', ms: 0 }
  if (last.state === 'succeeded' && rows.length > 1) return { state: 'retried', ms: last.ms }
  return { state: last.state, ms: last.ms }
}

export function TaskStage() {
  const { id } = useParams()
  const navigate = useNavigate()
  const { found, looking } = useTask(id)
  const state = useStore((s) => s)
  const act = useStore((s) => s.act)
  const setUi = useStore((s) => s.setUi)
  if (!found) return looking ? null : <TasksStage />
  const { t, computeId, c } = found
  const nodes = nodesOf(state, computeId)
  const rows = execsOf(t, nodes)
  const byRank = new Map<number, ExecRow[]>()
  for (const row of rows) byRank.set(row.rank, [...(byRank.get(row.rank) ?? []), row])

  const cells = holdersOf(nodes).map((n) => {
    const { state: what, ms: took } = outcome(byRank.get(n.rank) ?? [])
    return {
      rank: n.rank,
      fill: EXEC[what] ?? EXEC.missing!,
      tip: `rank ${n.rank} · ${what === 'missing' ? 'not sent' : what}${took ? ` · ${dur(took)}` : ''}`,
    }
  })
  const kinds = [...byRank.values()].map(outcome)
  const tally = (what: string) => kinds.filter((k) => k.state === what).length
  const not = cells.length - kinds.length
  const lens = fillRoom(cells.length, LENS[0], LENS[1])

  const timed = kinds.filter((k) => k.ms > 0).map((k) => k.ms).sort((a, b) => a - b)
  const fastest = timed[0] ?? 0
  const slowest = timed[timed.length - 1] ?? 0
  const step = (slowest - fastest) / BINS || 1
  const bins = Array.from({ length: BINS }, (_, i) => timed.filter((v) => v >= fastest + i * step && (i === BINS - 1 ? true : v < fastest + (i + 1) * step)).length)
  const ranked = [...byRank.entries()].map(([rank, all]) => ({ rank, ...outcome(all) })).sort((a, b) => b.ms - a.ms)
  const failures = rows.filter((e) => e.state === 'failed')
  const ran = runOf(t)

  const cancel = async () => {
    await api.cancelTask(t.id)
    await useStore.getState().reloadCompute(computeId)
  }
  const again = async () => {
    await api.retry(t.id)
    await useStore.getState().reloadCompute(computeId)
  }

  return (
    <>
      <PageHead
        title={<Fn sha={t.function.sha256} />}
        state={<Pill state={t.state} />}
        facts={[
          <>
            on <button onClick={() => navigate(`/computes/${c.id}`)}>{c.name ?? c.id}</button>
          </>,
          <>
            <b>{t.dispatch === 'all' ? 'every node' : t.dispatch === 'stream' ? 'stream' : 'one node'}</b>, {ranksOf(t, nodes)} rank{ranksOf(t, nodes) === 1 ? '' : 's'}
          </>,
          `submitted ${ago(ms(t.submitted_at))}`,
          <>
            took <b>{ran === null ? '—' : dur(ran)}</b>
          </>,
          `${c.spec.worker?.executor ?? 'thread'} × ${slotsOf(c)}`,
          <span className="mono">{t.id}</span>,
        ]}
        primary={
          t.state === 'running' || t.state === 'queued'
            ? { label: 'Cancel', icon: 'close', onClick: () => void cancel() }
            : { label: 'Run again', icon: 'refresh', onClick: () => void again() }
        }
        rest={[
          {
            label: 'Logs',
            onClick: () => {
              const only = [...byRank.keys()]
              setUi({ act: { ...act, kind: 'logs', compute: computeId, rank: only.length === 1 ? only[0]! : 'all' } })
              navigate('/activity')
            },
          },
        ]}
      />

      <section className="card tk" style={{ '--map': `${lens.width}px` }}>
        <div className="map">
          <span className="h">Ranks</span>
          {cells.length ? (
            <Hive
              cells={cells}
              computeId={`${computeId}/exec`}
              size={lens.size}
              label={`${cells.length} ranks`}
              onPick={(rank) => navigate(`/computes/${computeId}/nodes/${rank}`)}
            />
          ) : null}
          <div className="legend">
            {tally('succeeded') ? (
              <span>
                <i style={{ background: EXEC.succeeded }} />
                <b>{tally('succeeded')}</b> succeeded
              </span>
            ) : null}
            {tally('retried') ? (
              <span>
                <i style={{ background: EXEC.retried }} />
                <b>{tally('retried')}</b> retried
              </span>
            ) : null}
            {tally('started') ? (
              <span>
                <i style={{ background: EXEC.started }} />
                <b>{tally('started')}</b> running
              </span>
            ) : null}
            {tally('failed') ? (
              <span>
                <i style={{ background: EXEC.failed }} />
                <b>{tally('failed')}</b> failed
              </span>
            ) : null}
            {not > 0 ? (
              <span>
                <i style={{ background: EXEC.missing }} />
                <b>{not}</b> not sent
              </span>
            ) : null}
          </div>
        </div>
        <div className="dist">
          <span className="h">Duration per rank</span>
          {timed.length ? (
            <>
              <div className="row wrap" style={{ gap: 32 }}>
                <span className="fig">
                  <span className="num">{dur(fastest)}</span>
                  <span className="l">fastest</span>
                </span>
                <span className="fig">
                  <span className="num">{dur(median(timed))}</span>
                  <span className="l">median</span>
                </span>
                <span className="fig">
                  <span className="num">{dur(slowest)}</span>
                  <span className="l">slowest{ranked[0] ? `, rank ${ranked[0].rank}` : ''}</span>
                </span>
              </div>
              <Histo values={bins} mark={bins.length - 1} labels={[dur(fastest), dur(slowest)]} />
              {ranked.length > 3 ? (
                <div className="row wrap" style={{ gap: 6 }}>
                  <span className="sub" style={{ marginRight: 2 }}>
                    Slowest
                  </span>
                  {ranked.slice(0, 4).map((r) => (
                    <button key={r.rank} className="pill" onClick={() => navigate(`/computes/${computeId}/nodes/${r.rank}`)}>
                      rank {r.rank} <b>{dur(r.ms)}</b>
                    </button>
                  ))}
                </div>
              ) : null}
            </>
          ) : (
            <span className="sub">Nothing has finished on a rank yet.</span>
          )}
        </div>
      </section>

      {failures.length ? (
        <section className="card">
          <div className="chead" style={{ marginBottom: 2 }}>
            <span className="h">Failures</span>
            <span className="sub">{failures.length}</span>
          </div>
          {failures.map((e) => {
            const after = (byRank.get(e.rank) ?? []).find((r) => r.ordinal > e.ordinal)
            return (
              <div className="fail" key={`${e.rank}/${e.ordinal}`}>
                <b>rank {e.rank}</b>
                <span className="sub">attempt {e.ordinal}</span>
                <span className="sub">{dur(e.ms)}</span>
                <span className="mono" style={{ color: 'var(--bad)' }}>
                  {e.error}
                </span>
                {after ? <Pill state={after.state === 'succeeded' ? 'succeeded' : after.state} /> : <span className="sub">not retried</span>}
              </div>
            )
          })}
        </section>
      ) : null}

      <ClusterLoad c={c} from={ms(t.submitted_at)} to={t.finished_at ? ms(t.finished_at) : null} />
    </>
  )
}

/** What the whole compute's machines were doing while this task ran: the median, with the band from the lowest node to the highest. */
function ClusterLoad({ c, from, to }: { c: Compute; from: number; to: number | null }) {
  const span = spanOver(from, to)
  const feed = useMetrics(c.id, span)
  const accelerated = acceleratedOf(c)
  const marks = acrossNodes(feed, loadGauge(accelerated))
  if (!marks.line.length) return null
  return (
    <section className="card">
      <div className="chead" style={{ marginBottom: 10 }}>
        <span className="h">{accelerated ? 'Accelerator' : 'CPU'} across the cluster while it ran</span>
        <span className="sub spread">median, with the lowest and the highest node</span>
      </div>
      <Plot marks={marks} axis={axisOf(feed)} w={1080} h={64} />
    </section>
  )
}
