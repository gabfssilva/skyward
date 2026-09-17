import { useEffect, useRef, useState, type ReactNode } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import type { Compute, Node, Task } from '../../api/client'
import { useStore, historyOf, computeById, isLive, useLogs, useEvents, spentOf } from '../../state/store'
import type { Store } from '../../state/store'
import { CAUSE, HOUR, UNIT, ago, busyOf, callsOf, dateOf, dur, endedAt, execsOf, failedOf, finishedOf, hiveSize, median, money, ms, offerPerGpu, perGpu, ranOf, rateOf, readyOf, slotsOf, targetOf } from '../../state/model'
import type { MetricKey } from '../../state/model'
import { combNodes, valuesFor } from '../../state/nodes'
import { dispatchLine } from '../tasks/Stage'
import { Fn, Legend, Pill, TableScroll } from '../../ui/primitives'
import { Comb } from '../../ui/comb'
import { Spark } from '../../ui/charts'
import { Icon } from '../../ui/icons'
import { EvLineRow, LogLineRow } from '../../ui/lines'
import { usePorts } from '../../sheets/port-state'

const NONE: never[] = []
const SERIES = 32

/** A second hand for the page: what says `up 3h 43m` and `renews in 42s` moves without a store change. */
function useTick(on: boolean): void {
  const [, bump] = useState(0)
  useEffect(() => {
    if (!on) return
    const id = setInterval(() => bump((n) => n + 1), 1000)
    return () => clearInterval(id)
  }, [on])
}

export function Stage() {
  const { id = '' } = useParams()
  const c = useStore((s) => computeById(s, id))
  const live = useStore((s) => isLive(s, id))
  const nodes = useStore((s) => s.nodes[id]) ?? NONE
  const tasks = useStore((s) => s.tasks[id]) ?? NONE
  const loaded = nodes.length > 0
  useTick(live)
  useEffect(() => {
    if (!live && c && !loaded) void useStore.getState().reloadHistory(id)
  }, [live, c, loaded, id])
  /* history is paged, so the compute this page is about can be one nothing loaded */
  useEffect(() => {
    if (!c) void useStore.getState().learn([id])
  }, [c, id])
  if (!c) return null

  const name = c.name ?? c.id
  const when = live
    ? `up ${dur(Date.now() - ms(c.created_at))} · generation ${c.generation}${c.lease.owner ? ` · held by ${c.lease.owner}` : ''}`
    : `created ${dateOf(ms(c.created_at))} · ended ${c.ended ? ago(endedAt(c)) : '—'}`

  return (
    <>
      <section className="card cmp">
        <header className="cmp-id">
          <div className="row wrap" style={{ gap: 10 }}>
            <h2>{name}</h2>
            <Pill state={c.status.state} />
            <span className="mono faint">{c.id}</span>
            <span className="sub" style={{ marginLeft: 'auto' }}>
              {when}
            </span>
          </div>
          <div className="sub cmp-shape">
            <ShapeLine c={c} nodes={nodes} live={live} />
          </div>
        </header>
        <div className="cmp-main">{live ? <LiveBody c={c} nodes={nodes} tasks={tasks} /> : <EndedBody c={c} nodes={nodes} />}</div>
        <footer className="cmp-foot">
          <SpecGrid c={c} live={live} />
        </footer>
      </section>
      <section className="card">
        <TasksBlock c={c} tasks={tasks} nodes={nodes} />
      </section>
      <LogStream computeId={id} live={live} />
      <section className="card">
        <EventsBlock computeId={id} name={name} />
      </section>
    </>
  )
}

/* ---------- header ---------- */

function ShapeLine({ c, nodes, live }: { c: Compute; nodes: readonly Node[]; live: boolean }) {
  const s0 = c.spec.specs[0]
  const o = c.offer
  const b = c.spec.nodes
  const floor = b.min ?? b.initial
  const size = b.max ? `${floor}–${b.max} nodes, elastic` : floor !== b.initial ? `${b.initial} nodes, floor ${floor}` : `${live ? b.initial : nodes.length || b.initial} nodes`
  const terms = `${c.spec.allocation.replace(/_/g, ' ')}, ${c.spec.selection}`
  if (o)
    return (
      <>
        {o.accelerator ? (
          <>
            <b>
              {o.accelerator_count}× {o.accelerator.toUpperCase()}
            </b>{' '}
          </>
        ) : null}
        {o.instance_type} on <b>{o.kind}</b> {o.region ?? 'any region'} · {size} · {terms}
      </>
    )
  if (!s0) return <>{size}</>
  return (
    <>
      {s0.accelerator ? (
        <>
          <b>
            {s0.accelerator_count}× {s0.accelerator.toUpperCase()}
          </b>{' '}
          on{' '}
        </>
      ) : null}
      <b>{s0.provider.kind}</b> {s0.region ?? 'any region'} · {size} · {terms}
    </>
  )
}

/* ---------- figures ---------- */

function Fig({ value, unit, label, sub, spark, err }: { value: ReactNode; unit?: string; label: string; sub: string; spark?: ReactNode; err?: boolean }) {
  return (
    <div className="fig">
      <span className="fig-l">{label}</span>
      <b className={err ? 'err' : undefined}>
        {value}
        {unit ? <small>{unit}</small> : null}
      </b>
      {spark}
      <span className="fig-s">{sub}</span>
    </div>
  )
}

const Figs = ({ children }: { children: ReactNode }) => <div className="figs">{children}</div>

/** The cluster's median of one metric, sample by sample, over the ready nodes' histories. */
const clusterSeries = (s: Store, computeId: string, ready: readonly Node[], metric: MetricKey): number[] => {
  if (!ready.length) return []
  const series = ready.map((n) => historyOf(s, computeId, n.rank, metric))
  return Array.from({ length: SERIES }, (_, i) => median(series.map((h) => h[i] ?? h[h.length - 1] ?? 0)))
}

function LiveBody({ c, nodes, tasks }: { c: Compute; nodes: readonly Node[]; tasks: readonly Task[] }) {
  const navigate = useNavigate()
  const state = useStore((s) => s)
  const metrics = state.metrics
  const id = c.id
  const ready = readyOf(nodes)
  const sl = slotsOf(c)
  const busy = ready.reduce((s, n) => s + busyOf(tasks, nodes, n.rank), 0)

  const use = (k: MetricKey, label: string) => {
    const v = valuesFor(id, nodes, metrics, k)
    return (
      <Fig
        key={k}
        value={Math.round(median(v) || 0)}
        unit={UNIT[k]}
        label={label}
        sub={v.length ? `${Math.round(Math.min(...v))}–${Math.round(Math.max(...v))}${UNIT[k]} on ${v.length} nodes` : 'no node is ready'}
        spark={<Spark values={clusterSeries(state, id, ready, k)} h={30} fmt={(x) => Math.round(x) + UNIT[k]} />}
      />
    )
  }

  const running = tasks.filter((t) => t.state === 'running')
  const inflight = running.reduce((s, t) => s + execsOf(t, nodes).filter((e) => e.state === 'started').length, 0)
  const latest = (states: readonly Task['state'][]) => tasks.filter((t) => states.includes(t.state)).sort((a, b) => ms(b.finished_at) - ms(a.finished_at))[0]
  const lastDone = latest(['succeeded'])
  const lastErr = latest(['failed', 'timed_out'])
  const failed = failedOf(c)
  const fnName = (sha: string) => state.functions[sha]?.name ?? sha.slice(0, 8)
  const ranksOf = (t: Task) => new Set(execsOf(t, nodes).map((e) => e.rank)).size
  const cells = combNodes(id, nodes, metrics, state.progress)

  return (
    <>
      <div className="cmp-comb">
        <Comb
          layout="hive"
          nodes={cells}
          computeId={id}
          name={c.name ?? c.id}
          size={Math.min(60, hiveSize(cells.length, 500, 460))}
          onPick={(rank) => navigate(`/computes/${id}/nodes/${rank}`)}
        />
        <div className="row wrap" style={{ gap: 14 }}>
          <Legend states={cells.map((n) => n.state)} />
          <span className="mono faint">
            {sl > 1 ? `${busy} of ${ready.length * sl} slots busy · ${c.spec.worker?.executor ?? 'thread'} × ${sl}` : `${busy} of ${ready.length} busy`}
          </span>
        </div>
      </div>
      <div className="cmp-figs">
        <Figs>
          {use('gpu', 'Accelerator')}
          {use('vram', 'Memory')}
          {use('cpu', 'CPU')}
        </Figs>
        <Figs>
          <Fig
            value={inflight}
            label="in flight"
            sub={running.length ? running.map((t) => `${fnName(t.function)} on ${t.dispatch === 'one' ? 'one node' : `${ranksOf(t)} nodes`}`).join(', ') : 'nothing is running'}
          />
          <Fig value={c.tasks.succeeded} label="tasks done" sub={lastDone ? `${fnName(lastDone.function)}, ${ago(ms(lastDone.finished_at))}` : c.tasks.succeeded ? '—' : 'none yet'} />
          <Fig value={failed} label="errors" sub={lastErr ? `${fnName(lastErr.function)}, ${ago(ms(lastErr.finished_at))}` : failed ? '—' : 'none'} err={failed > 0} />
        </Figs>
        <BillFigs c={c} nodes={nodes} live />
      </div>
    </>
  )
}

function EndedBody({ c, nodes }: { c: Compute; nodes: readonly Node[] }) {
  const bought = c.offer ?? c.spec.specs[0]
  const ran = ranOf(c)
  const calls = callsOf(c)
  const failed = failedOf(c)
  return (
    <>
      <div className="cmp-out">
        <span className="fig-l">{ran < 60e3 ? 'Never became ready' : `Ran ${dur(ran)}`}</span>
        <b>Deleted</b>
        <span>{c.ended ? CAUSE[c.ended.cause] : '—'}</span>
      </div>
      <div className="cmp-figs">
        <Figs>
          <Fig value={nodes.length || targetOf(c)} label="nodes at peak" sub={bought?.accelerator ? `${bought.accelerator_count}× ${bought.accelerator.toUpperCase()} each` : (c.offer?.instance_type ?? '—')} />
          <Fig value={calls || '—'} label="calls" sub={calls ? `${Math.round(calls / (ran / HOUR))} an hour` : 'nothing ran here'} />
          <Fig value={failed} label="failed" sub={failed ? `${Math.round((failed / calls) * 1000) / 10}% of calls` : 'none'} err={failed > 0} />
        </Figs>
        <BillFigs c={c} nodes={nodes} live={false} />
      </div>
    </>
  )
}

/**
 * The bill: accrued from the nodes' rates while the compute is live, the daemon's
 * closed total once it has ended. The price per card comes from the nodes, and is
 * compared with the cheapest offer of the same accelerator.
 */
function BillFigs({ c, nodes, live }: { c: Compute; nodes: readonly Node[]; live: boolean }) {
  const offers = useStore((s) => s.offers)
  const s0 = c.spec.specs[0]
  const kind = c.offer?.kind ?? s0?.provider.kind
  const accelerator = c.offer?.accelerator ?? s0?.accelerator
  const calls = live ? finishedOf(c) : callsOf(c)
  const spent = useStore((s) => spentOf(s, c)) ?? 0
  const up = live ? Date.now() - ms(c.created_at) : ranOf(c)
  const hourly = live ? rateOf(nodes) : spent / (up / HOUR)
  const paid = perGpu(c, nodes)
  const cheap = offers
    .filter((o) => s0 && o.accelerator === accelerator)
    .map((o) => ({ o, per: offerPerGpu(o) }))
    .filter((x) => x.per > 0)
    .sort((a, b) => a.per - b.per)[0]
  return (
    <Figs>
      <Fig
        value={live ? (spent ? money(spent, 0) : '—') : money(spent, spent < 10 ? 2 : 0)}
        label={live ? 'spent so far' : 'spent in total'}
        sub={calls && spent ? `${money(spent / calls, spent / calls < 10 ? 2 : 0)} ${live ? 'per finished call' : 'per call'}` : live ? 'no finished call yet' : '—'}
      />
      <Fig
        value={live ? (hourly ? money(hourly, 0) : '—') : money(hourly, hourly < 10 ? 2 : 0)}
        unit="/h"
        label={live ? 'burning' : 'burned while up'}
        sub={live ? `up ${dur(up)}` : up < 60e3 ? 'never became ready' : `over ${dur(up)}`}
      />
      <Fig
        value={paid ? money(paid) : '—'}
        unit="/GPU·h"
        label={`paid on ${kind ?? '—'}`}
        sub={cheap && paid ? (cheap.per * 1.05 < paid ? `${(paid / cheap.per).toFixed(1)}× the cheapest (${money(cheap.per)}, ${cheap.o.kind})` : 'single offer price') : 'no offer to compare'}
      />
    </Figs>
  )
}

/* ---------- spec ---------- */

function SpecItem({ label, children, wide }: { label: string; children: ReactNode; wide?: 'wide' | 'wide3' }) {
  return (
    <div className={`spec-i${wide ? ` ${wide}` : ''}`}>
      <span>{label}</span>
      <span className="mono">{children}</span>
    </div>
  )
}

const Pkgs = ({ xs }: { xs: readonly string[] }) =>
  xs.length ? (
    <>
      {xs.slice(0, 3).join(', ')}
      {xs.length > 3 ? <span className="faint"> +{xs.length - 3}</span> : null}
    </>
  ) : (
    <span className="faint">none</span>
  )

function SpecGrid({ c, live }: { c: Compute; live: boolean }) {
  const openSheet = useStore((s) => s.openSheet)
  const ports = usePorts((s) => s.ports[c.id]) ?? NONE
  const im = c.spec.image
  const pip = im?.pip ?? NONE
  const apt = im?.apt ?? NONE
  const lease = Math.max(0, Math.round((ms(c.lease.expires_at) - Date.now()) / 1000))
  return (
    <>
      <div className="cap">Spec</div>
      <div className="specgrid">
        {im?.base ? (
          <SpecItem label="container" wide={im.base.length > 28 ? 'wide3' : 'wide'}>
            {im.base}
          </SpecItem>
        ) : null}
        <SpecItem label="python">{im?.python ?? <span className="faint">the machine's</span>}</SpecItem>
        <SpecItem label="pip" wide={pip.slice(0, 3).join(', ').length > 20 ? 'wide' : undefined}>
          <Pkgs xs={pip} />
        </SpecItem>
        {apt.length ? (
          <SpecItem label="apt">
            <Pkgs xs={apt} />
          </SpecItem>
        ) : null}
        <SpecItem label="executor">
          {c.spec.worker?.executor ?? 'thread'} × {slotsOf(c)}
        </SpecItem>
        <SpecItem label="plugins">{c.spec.plugins.map((p) => p.kind).join(', ') || <span className="faint">none</span>}</SpecItem>
        {live ? (
          <>
            <SpecItem label="lease">{c.lease.expires_at ? `renews in ${lease}s` : '—'}</SpecItem>
            <SpecItem label="ports" wide="wide">
              {ports.map((p) => `${p.remote}→${p.local}`).join(', ') || 'none'}
              <button
                className="btn sm ghost"
                style={{ height: 20, padding: '0 5px', fontSize: 11, marginLeft: 4 }}
                onClick={() => openSheet({ kind: 'ports', computeId: c.id })}
              >
                <Icon name="ports" />
                Forward
              </button>
            </SpecItem>
          </>
        ) : null}
      </div>
    </>
  )
}

/* ---------- tasks ---------- */

const attemptOf = (t: Task): number => t.executions.reduce((n, e) => Math.max(n, e.ordinal), 1)

function TasksBlock({ c, tasks, nodes }: { c: Compute; tasks: readonly Task[]; nodes: readonly Node[] }) {
  const navigate = useNavigate()
  const setUi = useStore((s) => s.setUi)
  const ended = c.ended
  const list = tasks.slice().sort((a, b) => ms(b.submitted_at) - ms(a.submitted_at))
  const shown = list.slice(0, 8)
  const calls = callsOf(c)
  const failed = failedOf(c)
  /* the daemon keeps every task: what this card leaves out is in the Tasks view, filtered to this compute */
  const more = calls - shown.length
  return (
    <div>
      <div className="combhead">
        <b>Tasks</b>
        <span className="sub">
          {!calls
            ? 'nothing ran here'
            : ended
              ? `showing the newest ${shown.length} of ${calls} calls${failed ? `, ${failed} failed` : ''}`
              : `${c.tasks.running} running · ${c.tasks.queued} queued · ${failed} failed · ${c.tasks.succeeded} succeeded`}
        </span>
      </div>
      {list.length ? (
        <>
          <TableScroll label="Tasks">
            <table>
              <thead>
                <tr>
                  <th>Function</th>
                  <th>State</th>
                  <th>Submitted</th>
                  <th className="right">Took</th>
                </tr>
              </thead>
              <tbody>
                {shown.map((t) => {
                  const attempt = attemptOf(t)
                  return (
                    <tr key={t.id} style={{ cursor: 'pointer' }} onClick={() => navigate(`/tasks/${t.id}`)}>
                      <td>
                        <Fn sha={t.function} weight={700} />
                        <div className="mono faint">
                          {dispatchLine(t, nodes)}
                          {attempt > 1 ? ` · attempt ${attempt}` : ''}
                        </div>
                      </td>
                      <td>
                        <Pill state={t.state} />
                      </td>
                      <td className="mono faint" style={{ whiteSpace: 'nowrap' }}>
                        {ago(ms(t.submitted_at))}
                      </td>
                      <td className="right mono" style={{ whiteSpace: 'nowrap' }}>
                        {dur((t.finished_at ? ms(t.finished_at) : Date.now()) - ms(t.submitted_at))}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </TableScroll>
          {more > 0 ? (
            <button
              className="btn sm ghost"
              style={{ marginTop: 8 }}
              onClick={() => {
                setUi({ task: { compute: c.id, state: 'all' } })
                navigate('/tasks')
              }}
            >
              <Icon name="tasks" />
              {more} more in Tasks
            </button>
          ) : null}
        </>
      ) : null}
    </div>
  )
}

/* ---------- what it prints and what happened ---------- */

/** The last dozen lines the compute printed, following as they arrive while it is still printing. */
function LogStream({ computeId, live }: { computeId: string; live: boolean }) {
  const navigate = useNavigate()
  const feed = useLogs({ compute: computeId })
  const act = useStore((s) => s.act)
  const setUi = useStore((s) => s.setUi)
  const box = useRef<HTMLDivElement>(null)
  const tail = (feed?.lines ?? NONE).slice(-12)
  useEffect(() => {
    if (box.current) box.current.scrollTop = box.current.scrollHeight
  }, [tail[tail.length - 1]?.sequence])
  return (
    <section className="card">
      <div className="combhead">
        <b>{live ? 'Log stream' : 'What it printed'}</b>
        <button
          className="btn sm ghost"
          style={{ marginLeft: 'auto' }}
          onClick={() => {
            setUi({ act: { ...act, kind: 'logs', compute: computeId, rank: 'all' } })
            navigate('/activity')
          }}
        >
          <Icon name="logs" />
          Open in Activity
        </button>
      </div>
      <div className="logbox short flat" ref={box}>
        {tail.length ? tail.map((l) => <LogLineRow key={`${l.sequence}.${l.part}`} line={l} />) : <div className="sub">Nothing printed yet.</div>}
      </div>
    </section>
  )
}

function EventsBlock({ computeId, name }: { computeId: string; name: string }) {
  const navigate = useNavigate()
  const evs = useEvents(computeId).items.slice(0, 6)
  const act = useStore((s) => s.act)
  const setUi = useStore((s) => s.setUi)
  return (
    <div>
      <div className="combhead">
        <b>Events</b>
        <button
          className="btn sm ghost"
          style={{ marginLeft: 'auto' }}
          onClick={() => {
            setUi({ act: { ...act, kind: 'events', compute: computeId, rank: 'all' } })
            navigate('/activity')
          }}
        >
          <Icon name="events" />
          Open in Activity
        </button>
      </div>
      <div className="evbox">{evs.length ? evs.map((e) => <EvLineRow key={e.id} event={e} computeName={name} />) : <div className="sub">Nothing happened yet.</div>}</div>
    </div>
  )
}
