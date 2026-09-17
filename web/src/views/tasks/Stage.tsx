import { useNavigate } from 'react-router-dom'
import type { Node, Task } from '../../api/client'
import { computeById, freshest, useStore, useTasks } from '../../state/store'
import type { TaskFilters } from '../../state/store'
import { ago, dur, ms, readyOf } from '../../state/model'
import { Fn, Pill, TableScroll } from '../../ui/primitives'

const NONE: never[] = []

const attemptOf = (t: Task): number => t.executions.reduce((n, e) => Math.max(n, e.ordinal), 1)

/** How many ranks a task reached: its recorded executions, or, before any is recorded, the nodes it was sent to. */
export const ranksOf = (t: Task, nodes: readonly Node[] = []): number => new Set(t.executions.map((e) => e.rank)).size || readyOf(nodes).length

export const dispatchLine = (t: Task, nodes: readonly Node[] = []): string =>
  t.dispatch === 'all' ? `every node · ${ranksOf(t, nodes)}` : t.dispatch === 'stream' ? `stream · ${ranksOf(t, nodes)}` : 'one node'

/** The states a task can be in, as the daemon's filter takes them. */
const STATES: readonly (readonly [TaskFilters['state'], string])[] = [
  ['all', 'any state'],
  ['queued', 'queued'],
  ['running', 'running'],
  ['succeeded', 'succeeded'],
  ['failed', 'failed'],
  ['cancelled', 'cancelled'],
  ['timed_out', 'timed out'],
  ['indeterminate', 'indeterminate'],
]

/** The state a select's value names, so what the daemon is asked for is one of its own, without a cast. */
const stateOf = (value: string): TaskFilters['state'] => STATES.find(([state]) => state === value)?.[0] ?? 'all'

/** One task of the feed: the freshest copy the store has of it, under its compute's id until the store learns the name. */
function Row({ task }: { task: Task }) {
  const navigate = useNavigate()
  const t = useStore((s) => freshest(s, task))
  const c = useStore((s) => computeById(s, task.compute_id))
  const nodes = useStore((s) => s.nodes[task.compute_id]) ?? NONE
  return (
    <tr style={{ cursor: 'pointer' }} onClick={() => navigate(`/tasks/${t.id}`)}>
      <td>
        <Fn sha={t.function} weight={700} />
        <div className="mono faint">{t.id}</div>
      </td>
      <td>
        <Pill state={t.state} />
      </td>
      <td>
        <span className="row" style={{ gap: 6 }}>
          <i className={c ? `dot ${c.status.state}` : 'dot'} />
          <span className={c ? undefined : 'mono faint'}>{c?.name ?? task.compute_id}</span>
        </span>
      </td>
      <td className="sub">{dispatchLine(t, nodes)}</td>
      <td className="mono">{attemptOf(t)}</td>
      <td className="mono faint">{ago(ms(t.submitted_at))}</td>
      <td className="right mono">{dur((t.finished_at ? ms(t.finished_at) : Date.now()) - ms(t.submitted_at))}</td>
    </tr>
  )
}

/** The Tasks view: one page of what the daemon matched under ``Ui.task``, with older pages asked for on request. */
export function Stage() {
  const feed = useTasks()
  const computes = useStore((s) => s.computes)
  const history = useStore((s) => s.history)
  const f = useStore((s) => s.task)
  const setUi = useStore((s) => s.setUi)
  const pageTasks = useStore((s) => s.pageTasks)
  const set = (patch: Partial<TaskFilters>) => setUi({ task: { ...f, ...patch } })
  const rows = feed?.items ?? NONE
  const total = feed?.total ?? null
  const running = rows.filter((t) => t.state === 'running').length
  const named = [...computes, ...history].some((c) => c.id === f.compute)
  return (
    <section className="card">
      <div className="combhead">
        <b>Tasks</b>
        <span className="sub">
          {total === null ? rows.length : `${rows.length} of ${total}`} · {running} running
        </span>
        <select
          className="search"
          aria-label="compute"
          style={{ minWidth: 150, marginLeft: 'auto' }}
          value={f.compute}
          onChange={(e) => set({ compute: e.target.value })}
        >
          <option value="all">every compute</option>
          {computes.map((c) => (
            <option key={c.id} value={c.id}>
              {c.name ?? c.id}
            </option>
          ))}
          {history.map((c) => (
            <option key={c.id} value={c.id}>
              {c.name ?? c.id} · ended
            </option>
          ))}
          {f.compute !== 'all' && !named ? <option value={f.compute}>{f.compute}</option> : null}
        </select>
        <select className="search" aria-label="state" style={{ minWidth: 130 }} value={f.state} onChange={(e) => set({ state: stateOf(e.target.value) })}>
          {STATES.map(([state, label]) => (
            <option key={state} value={state}>
              {label}
            </option>
          ))}
        </select>
      </div>
      <TableScroll label="Tasks">
        <table className="cards tasklist">
          <thead>
            <tr>
              <th>Function</th>
              <th>State</th>
              <th>Compute</th>
              <th>Dispatch</th>
              <th>Attempt</th>
              <th>Submitted</th>
              <th className="right">Took</th>
            </tr>
          </thead>
          <tbody>
            {rows.length ? (
              rows.map((t) => <Row key={t.id} task={t} />)
            ) : (
              <tr>
                <td colSpan={7} className="sub" style={{ padding: '22px 8px', textAlign: 'center' }}>
                  {(feed?.loading ?? true) ? 'Reading tasks…' : 'No task matches these filters.'}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </TableScroll>
      {feed?.cursor ? (
        <button className="btn sm" style={{ marginTop: 10 }} disabled={feed.loading} onClick={() => void pageTasks()}>
          Show older tasks
        </button>
      ) : null}
    </section>
  )
}
