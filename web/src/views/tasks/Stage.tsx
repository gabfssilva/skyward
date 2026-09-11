import { useNavigate } from 'react-router-dom'
import { useStore } from '../../state/store'
import type { Compute, Task } from '../../api/client'
import { ago, dur, ms } from '../../state/model'
import { Fn, Pill } from '../../ui/primitives'

const attemptOf = (t: Task): number => t.executions.reduce((n, e) => Math.max(n, e.ordinal), 1)
const ranksOf = (t: Task): number => new Set(t.executions.map((e) => e.rank)).size

export const dispatchLine = (t: Task): string =>
  t.dispatch === 'all' ? `every node · ${ranksOf(t)}` : t.dispatch === 'stream' ? `stream · ${ranksOf(t)}` : 'one node'

function Row({ t, c }: { t: Task; c: Compute }) {
  const navigate = useNavigate()
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
          <i className={`dot ${c.status.state}`} />
          {c.name}
        </span>
      </td>
      <td className="sub">{dispatchLine(t)}</td>
      <td className="mono">{attemptOf(t)}</td>
      <td className="mono faint">{ago(ms(t.submitted_at))}</td>
      <td className="right mono">{dur((t.finished_at ? ms(t.finished_at) : Date.now()) - ms(t.submitted_at))}</td>
    </tr>
  )
}

export function Stage() {
  const computes = useStore((s) => s.computes)
  const tasks = useStore((s) => s.tasks)
  const list = computes
    .flatMap((c) => (tasks[c.id] ?? []).map((t) => ({ t, c })))
    .sort((a, b) => ms(b.t.submitted_at) - ms(a.t.submitted_at))
  const running = list.filter((x) => x.t.state === 'running')
  return (
    <section className="card">
      <div className="combhead">
        <b>Tasks</b>
        <span className="sub">
          {running.length} running · {list.length - running.length} finished
        </span>
      </div>
      <div className="scroll">
        <table>
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
            {list.map(({ t, c }) => (
              <Row key={t.id} t={t} c={c} />
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
