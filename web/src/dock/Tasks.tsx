import { useNavigate } from 'react-router-dom'
import { useStore } from '../state/store'
import type { Compute, Task } from '../api/client'
import { Fn, Pill } from '../ui/primitives'
import { dur, ms, readyOf } from '../state/model'
import type { Node } from '../api/client'

const attemptOf = (t: Task): number => t.executions.reduce((n, e) => Math.max(n, e.ordinal), 1)

export function Tasks({ computeId }: { computeId: string | null }) {
  const navigate = useNavigate()
  const computes = useStore((s) => s.computes)
  const history = useStore((s) => s.history)
  const tasks = useStore((s) => s.tasks)
  const nodes = useStore((s) => s.nodes)

  const rows: { t: Task; c: Compute }[] = (computeId ? [...computes, ...history] : computes)
    .filter((c) => !computeId || c.id === computeId)
    .flatMap((c) => (tasks[c.id] ?? []).map((t) => ({ t, c })))
    .sort((a, b) => ms(b.t.submitted_at) - ms(a.t.submitted_at))

  const ranksOf = (c: Compute): number => readyOf((nodes[c.id] ?? []) as Node[]).length

  return (
    <table>
      <thead>
        <tr>
          <th>Task</th>
          <th>Function</th>
          <th>Compute</th>
          <th>Dispatch</th>
          <th>State</th>
          <th>Attempt</th>
          <th className="right">Took</th>
          <th>Executions</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(({ t, c }) => (
          <tr key={t.id} style={{ cursor: 'pointer' }} onClick={() => navigate(`/tasks/${t.id}`)}>
            <td className="mono">{t.id}</td>
            <td>
              <Fn sha={t.function} />
            </td>
            <td>
              <button
                className="mono"
                onClick={(e) => {
                  e.stopPropagation()
                  navigate(`/computes/${c.id}`)
                }}
              >
                {c.name ?? c.id}
              </button>
            </td>
            <td className="sub">
              {t.dispatch === 'all' ? `every node (${ranksOf(c)})` : t.dispatch === 'stream' ? 'stream' : 'one node'}
            </td>
            <td>
              <Pill state={t.state} />
            </td>
            <td className="mono">{attemptOf(t)}</td>
            <td className="right mono">{dur((t.finished_at ? ms(t.finished_at) : Date.now()) - ms(t.submitted_at))}</td>
            <td>
              {t.executions.length ? (
                t.executions.map((x) => (
                  <span
                    key={x.id}
                    className="chip"
                    style={{
                      height: 20,
                      fontSize: 10,
                      ...(x.state === 'failed' ? { color: 'var(--bad)', background: 'var(--bad-soft)' } : {}),
                    }}
                    data-tip={`${x.error?.message ?? 'returned a value'} · rank ${x.rank} · ${dur(
                      (x.finished_at ? ms(x.finished_at) : Date.now()) - ms(x.started_at),
                    )}`}
                  >
                    #{x.ordinal} rank {x.rank}
                  </span>
                ))
              ) : (
                <span className="faint mono">{ranksOf(c)} ranks, no retry</span>
              )}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
