import { useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { api } from '../../api/client'
import type { Compute, Node, Task } from '../../api/client'
import { useStore, computeById, isLive, useLogs } from '../../state/store'
import { busyOf, dur, execsOf, holderOf, holdersOf, hiveSize, machineOf, money, ms, nodeHeld, slotsOf } from '../../state/model'
import type { ExecRow } from '../../state/model'
import { Fn, Pill, TableScroll } from '../../ui/primitives'
import { Hive, Slots } from '../../ui/comb'
import { PageHead, Tabs } from '../../ui/head'
import { LogBox } from '../../ui/lines'
import { Metrics } from '../compute/Metrics'
import { PhaseProgress } from '../compute/Progress'
import { Shell } from './Shell'
import { VRAM } from './Shell'
import { Stage as ComputeStage } from '../compute/Stage'

const NONE: never[] = []

type Tab = 'logs' | 'tasks' | 'shell'

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

/** Replacing a node is draining it: the reconciler buys the one that fills the gap. */
const replace = async (c: Compute, n: Node): Promise<void> => {
  await api.drainNode(c.id, n.id)
  await useStore.getState().reloadCompute(c.id)
}

/**
 * One machine: where it sits, what it is measuring, and a terminal on it.
 *
 * Every fact appears once. The node's own metrics carry the compute's median dashed behind them, which is
 * what the compute page's list of stragglers was for, and where the node sits in its compute is one small
 * hive instead of a card describing it.
 */
export function Stage() {
  const navigate = useNavigate()
  const found = useNode()
  const wanted = useStore((s) => s.shell)
  const setUi = useStore((s) => s.setUi)
  const [tab, setTab] = useState<Tab | null>(null)
  if (!found) return <ComputeStage />
  const { c, n, live, nodes, tasks } = found
  const ready = n.state === 'ready'
  const held = nodeHeld(n)
  const slots = slotsOf(c)
  const accel = n.accelerator ?? c.spec.specs[0]?.accelerator ?? null
  const vram = accel ? VRAM[accel] : undefined
  const at = tab ?? (wanted && held && live ? 'shell' : 'logs')

  const ran = tasks
    .map((t) => ({ t, x: execsOf(t, nodes).find((e) => e.rank === n.rank) }))
    .filter((o): o is { t: Task; x: ExecRow } => o.x != null)
    .slice(0, 10)
    .reverse()

  const drain = async () => {
    await api.drainNode(c.id, n.id)
    await useStore.getState().reloadCompute(c.id)
    navigate(`/computes/${c.id}`)
  }

  return (
    <>
      <PageHead
        crumb={{ label: c.name ?? c.id, onClick: () => navigate(`/computes/${c.id}`) }}
        title={`rank ${n.rank}`}
        state={<Pill state={n.state} />}
        id={n.id}
        why={n.last_error ? { message: n.last_error.message } : null}
        facts={[
          n.address ? <b>{n.address}</b> : null,
          <>
            <b>{machineOf(c)}</b>
            {vram ? `, ${vram} GB each` : ''}
          </>,
          <>
            <b>{money(n.price_per_hour ?? 0)}/h</b> {n.market === 'spot' ? 'spot' : 'on demand'}
          </>,
          ready ? `up ${dur(Date.now() - (ms(n.launched_at) || ms(n.created_at)))}` : null,
          `generation ${n.generation}`,
          n.machine ? <span className="mono">{n.machine}</span> : null,
        ]}
        primary={live && held ? { label: 'Shell', icon: 'shell', onClick: () => setTab('shell') } : undefined}
        rest={
          live
            ? [
                { label: 'Replace', icon: 'refresh', onClick: () => void replace(c, n) },
                { label: 'Drain', icon: 'drain', danger: true, onClick: () => void drain() },
              ]
            : undefined
        }
      />

      <section className="card">
        {ready ? (
          <div className="spot">
            <div className="row" style={{ gap: 16 }}>
              <span className="h" style={{ width: 44 }}>
                Slots
              </span>
              <Slots slots={slots} busy={busyOf(tasks, nodes, n.rank)} />
              <span className="sub">
                {busyOf(tasks, nodes, n.rank)} of {slots} busy, {c.spec.worker?.executor ?? 'thread'} × {slots}
              </span>
            </div>
            <div className="row" style={{ gap: 16, justifyContent: 'flex-end' }}>
              <span className="sub">
                rank {n.rank} of {holdersOf(nodes).length} in <b style={{ color: 'var(--ink)', fontWeight: 600 }}>{c.name ?? c.id}</b>
              </span>
              <Hive
                cells={holdersOf(nodes).map((other) => ({
                  rank: other.rank,
                  fill: other.rank === n.rank ? 'var(--accent)' : 'var(--sunk)',
                  tip: `rank ${other.rank} · ${other.state}`,
                }))}
                computeId={`${c.id}/locate`}
                size={Math.min(14, hiveSize(holdersOf(nodes).length, 190, 120))}
                label={`rank ${n.rank} of ${holdersOf(nodes).length}`}
                onPick={(rank) => navigate(`/computes/${c.id}/nodes/${rank}`)}
              />
            </div>
          </div>
        ) : (
          <PhaseProgress nodeId={n.id} />
        )}
      </section>

      {ready ? <Metrics computeId={c.id} name={c.name ?? c.id} nodes={nodes} created={ms(c.created_at)} node={n} /> : null}

      <section className="card">
        <Tabs<Tab>
          value={at}
          options={[
            ['logs', 'Logs', null],
            ['tasks', 'Tasks', ran.length || null],
            ['shell', 'Shell', null],
          ]}
          onChange={(next) => {
            setTab(next)
            if (next !== 'shell' && wanted) setUi({ shell: false })
          }}
        >
          <span className="sub spread">{at === 'shell' ? `pty over the daemon${n.address ? ` · ${n.address}` : ''}` : null}</span>
        </Tabs>
        <div className="tabbody">
          {at === 'shell' ? <Shell computeId={c.id} rank={n.rank} /> : null}
          {at === 'tasks' ? <RanHere rows={ran} /> : null}
          {at === 'logs' ? <RankLogs c={c} n={n} /> : null}
        </div>
      </section>
    </>
  )
}

function RanHere({ rows }: { rows: readonly { t: Task; x: ExecRow }[] }) {
  const navigate = useNavigate()
  if (!rows.length) return <div className="sub">Nothing has run on this node.</div>
  return (
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
            <tr key={`${t.id}/${x.ordinal}`} data-open="" onClick={() => navigate(`/tasks/${t.id}`)}>
              <td>
                <Fn sha={t.function.sha256} />
              </td>
              <td>
                <Pill state={x.state} />
                {x.error ? <span className="err-line">{x.error}</span> : null}
              </td>
              <td>{x.ordinal}</td>
              <td className="right nowrap">{dur(x.ms)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </TableScroll>
  )
}

/** What this node printed, scoped to it on the daemon: a quiet node keeps its own lines instead of being crowded out of the compute's window by a noisier peer. */
function RankLogs({ c, n }: { c: Compute; n: Node }) {
  const feed = useLogs({ compute: c.id, node: n.id })
  const pageLogs = useStore((s) => s.pageLogs)
  return (
    <LogBox
      lines={feed?.lines ?? NONE}
      ranks={false}
      older={
        feed?.cursor ? (
          <button className="btn sm" style={{ marginBottom: 10 }} disabled={feed.loading} onClick={() => void pageLogs()}>
            Load older
          </button>
        ) : null
      }
    />
  )
}
