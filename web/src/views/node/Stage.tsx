import { useEffect, useRef, type CSSProperties } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { api } from '../../api/client'
import type { Compute, Node, Task } from '../../api/client'
import { useStore, historyOf, computeById, isLive, useLogs } from '../../state/store'
import type { Store } from '../../state/store'
import { METRICS, UNIT, busyOf, dur, execsOf, hexPts, hive, holderOf, money, ms, slotsOf } from '../../state/model'
import type { ExecRow } from '../../state/model'
import type { PhaseMark } from '../../state/nodes'
import { Fn, Pill, TableScroll } from '../../ui/primitives'
import { Spark } from '../../ui/charts'
import { Icon } from '../../ui/icons'
import { LogLineRow } from '../../ui/lines'
import { ShellCard } from './Shell'
import { Stage as ComputeStage } from '../compute/Stage'

const NONE: never[] = []

/** The node the route names, or nothing when the compute has no such rank. */
export function useNode(): { c: Compute; n: Node; live: boolean; nodes: readonly Node[]; tasks: readonly Task[] } | null {
  const { id = '', rank = '0' } = useParams()
  const c = useStore((s) => computeById(s, id))
  const live = useStore((s) => isLive(s, id))
  const nodes = useStore((s) => s.nodes[id]) ?? NONE
  const tasks = useStore((s) => s.tasks[id]) ?? NONE
  const n = holderOf(nodes, Number(rank))
  return c && n ? { c, n, live, nodes, tasks } : null
}

const CHIP: Record<PhaseMark['state'], CSSProperties> = {
  completed: { color: 'var(--ok)', background: 'var(--ok-soft)' },
  started: { color: 'var(--boot)', background: 'var(--boot-soft)' },
  failed: { color: 'var(--bad)', background: 'var(--bad-soft)' },
}

/** What the phase and the progress of a node coming up read as: a fraction while the provider reports one, then the phases the machine has reached. */
export function PhaseProgress({ nodeId, maxWidth }: { nodeId: string; maxWidth?: number }) {
  const p = useStore((s) => s.progress[nodeId])
  return (
    <>
      <div className="cap">{p?.phase ?? 'waiting'}</div>
      {p?.completion != null ? (
        <div className="track" style={{ marginTop: 6, maxWidth }}>
          <i style={{ width: `${p.completion * 100}%`, background: 'var(--boot)' }} />
        </div>
      ) : null}
      {p?.phases.length ? (
        <div className="chips" style={{ marginTop: 9 }}>
          {p.phases.map((ph) => (
            <span key={ph.name} className="chip" style={{ height: 20, fontSize: 10, ...CHIP[ph.state] }}>
              {ph.state === 'completed' ? '✓ ' : ''}
              {ph.name}
            </span>
          ))}
        </div>
      ) : null}
    </>
  )
}

/** Replacing a node is draining it: the reconciler buys the one that fills the gap. */
export const replace = async (c: Compute, n: Node): Promise<void> => {
  await api.drainNode(c.id, n.id)
  await useStore.getState().reloadCompute(c.id)
}

export function Stage() {
  const found = useNode()
  const shell = useStore((s) => s.shell)
  const setUi = useStore((s) => s.setUi)
  if (!found) return <ComputeStage />
  const { c, n, live, nodes, tasks } = found
  const ready = n.state === 'ready'
  const ran = tasks.map((t) => ({ t, x: execsOf(t, nodes).find((e) => e.rank === n.rank) })).filter((o): o is { t: Task; x: ExecRow } => o.x != null).slice(0, 8)
  const sl = slotsOf(c)
  const meta = [
    n.address ?? 'no address',
    `${c.spec.specs[0]?.accelerator_count ?? 1}× ${(n.accelerator ?? c.spec.specs[0]?.accelerator ?? '?').toUpperCase()}`,
    `${money(n.price_per_hour ?? 0)}/h ${n.market === 'spot' ? 'spot' : 'on demand'}`,
    ready ? `up ${dur(Date.now() - (ms(n.launched_at) || ms(n.created_at)))}` : (n.last_error?.message ?? n.state),
  ].join(' · ')

  return (
    <>
      <section className="card">
        <div className="combhead" style={{ marginBottom: 16 }}>
          <span className="mono" style={{ background: 'var(--sunk)', padding: '3px 10px', borderRadius: 6, fontSize: 14 }}>
            rank {n.rank}
          </span>
          <b style={{ fontSize: 18 }}>{n.id}</b>
          <Pill state={n.state} />
          <span className="mono faint" style={{ marginLeft: 'auto' }}>
            {meta}
          </span>
        </div>
        {ready && sl > 1 ? (
          <div className="row" style={{ gap: 18, alignItems: 'flex-start', marginBottom: 16 }}>
            <SlotHive c={c} nodes={nodes} tasks={tasks} rank={n.rank} />
            <div className="sub" style={{ paddingTop: 6 }}>
              {busyOf(tasks, nodes, n.rank)} of {sl} slots busy · {c.spec.worker?.executor ?? 'thread'} × {sl}
            </div>
          </div>
        ) : null}
        {ready ? (
          <MetricGrid computeId={c.id} rank={n.rank} />
        ) : n.last_error ? (
          <>
            <div style={{ color: 'var(--bad)' }}>{n.last_error.message}</div>
            {live ? (
              <button className="btn sm" style={{ marginTop: 10 }} onClick={() => void replace(c, n)}>
                Replace it
              </button>
            ) : null}
          </>
        ) : (
          <PhaseProgress nodeId={n.id} maxWidth={420} />
        )}
      </section>
      {ran.length ? <RanHere rows={ran} /> : null}
      <RankLogs c={c} n={n} />
      {shell ? <ShellCard computeId={c.id} rank={n.rank} onClose={() => setUi({ shell: false })} /> : null}
    </>
  )
}

/** One hex per worker slot, lit while an execution occupies it. */
function SlotHive({ c, nodes, tasks, rank }: { c: Compute; nodes: readonly Node[]; tasks: readonly Task[]; rank: number }) {
  const k = slotsOf(c)
  const busy = busyOf(tasks, nodes, rank)
  const s = 30
  const lay = hive(k, s, 0.1)
  const P = hexPts(s)
  const running = tasks.filter((t) => t.state === 'running' && execsOf(t, nodes).some((e) => e.rank === rank && e.state === 'started'))
  return (
    <svg className="slothive" viewBox={`-1 -1 ${(lay.w + 2).toFixed(1)} ${(lay.h + 2).toFixed(1)}`} style={{ width: Math.ceil(lay.w), flex: 'none' }}>
      {lay.cells.map(([x, y], i) => {
        const t = i < busy ? running[i % Math.max(1, running.length)] : undefined
        return (
          <g key={i} transform={`translate(${x.toFixed(1)},${y.toFixed(1)})`} data-tip={t ? (t.function.name ?? t.function.sha256.slice(0, 8)) : 'idle'}>
            <polygon className={`slot${t ? ' on' : ''}`} points={P} />
            <text y="4">{i}</text>
          </g>
        )
      })}
    </svg>
  )
}

const metricsOf = (s: Store, computeId: string, rank: number) => s.metrics[`${computeId}/${rank}`]

function MetricGrid({ computeId, rank }: { computeId: string; rank: number }) {
  const state = useStore((s) => s)
  const m = metricsOf(state, computeId, rank)
  return (
    <div className="mgrid">
      {METRICS.map(([k, l]) => (
        <div className="mcard" key={k}>
          <span className="cap">{l}</span>
          <b>
            {Math.round(m?.[k] ?? 0)}
            <small>{UNIT[k]}</small>
          </b>
          <Spark values={historyOf(state, computeId, rank, k)} h={38} fmt={(v) => Math.round(v) + UNIT[k]} />
        </div>
      ))}
    </div>
  )
}

function RanHere({ rows }: { rows: readonly { t: Task; x: ExecRow }[] }) {
  const navigate = useNavigate()
  return (
    <section className="card">
      <div className="cap" style={{ marginBottom: 8 }}>
        Ran on this node
      </div>
      <TableScroll label="Ran on this node">
        <table>
          <thead>
            <tr>
              <th>Function</th>
              <th>State</th>
              <th>Attempt</th>
              <th className="right">Took</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(({ t, x }) => (
              <tr key={t.id} style={{ cursor: 'pointer' }} onClick={() => navigate(`/tasks/${t.id}`)}>
                <td>
                  <Fn sha={t.function.sha256} weight={600} />
                </td>
                <td>
                  <Pill state={x.state} />
                </td>
                <td className="mono">#{x.ordinal}</td>
                <td className="right mono">{dur(x.ms)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </TableScroll>
    </section>
  )
}

/** What this node printed, scoped to it on the daemon: a quiet node keeps its own lines instead of being crowded out of the compute's window by a noisier peer. */
function RankLogs({ c, n }: { c: Compute; n: Node }) {
  const feed = useLogs({ compute: c.id, node: n.id })
  const navigate = useNavigate()
  const act = useStore((s) => s.act)
  const setUi = useStore((s) => s.setUi)
  const pageLogs = useStore((s) => s.pageLogs)
  const box = useRef<HTMLDivElement>(null)
  const lines = feed?.lines ?? NONE
  useEffect(() => {
    if (box.current) box.current.scrollTop = box.current.scrollHeight
  }, [lines[lines.length - 1]?.sequence])
  return (
    <section className="card">
      <div className="row" style={{ justifyContent: 'space-between', marginBottom: 8 }}>
        <span className="cap">Logs of rank {n.rank}</span>
        <button
          className="btn sm"
          onClick={() => {
            setUi({ act: { ...act, kind: 'logs', compute: c.id, rank: n.rank } })
            navigate('/activity')
          }}
        >
          <Icon name="logs" />
          Open in Activity
        </button>
      </div>
      <div className="logbox" ref={box}>
        {lines.length ? lines.map((l) => <LogLineRow key={`${l.sequence}.${l.part}`} line={l} computeName={c.name ?? undefined} />) : <div className="sub">Nothing printed by this rank yet.</div>}
      </div>
      {feed?.cursor ? (
        <button className="btn sm" style={{ marginTop: 10 }} disabled={feed.loading} onClick={() => void pageLogs()}>
          Load older
        </button>
      ) : null}
    </section>
  )
}
