import { useEffect, useState, type ReactNode } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import type { Compute, Node, Task } from '../../api/client'
import { useStore, computeById, isLive, useLogs, useEvents, spentOf } from '../../state/store'
import {
  CAUSE,
  HOUR,
  acceleratedOf,
  ago,
  boundOf,
  busyOf,
  byState,
  callsOf,
  dateOf,
  dur,
  endedAt,
  execsOf,
  failedOf,
  finishedOf,
  machineOf,
  money,
  ms,
  perUnit,
  ranOf,
  rateOf,
  runOf,
  readyOf,
  sizeOf,
  slotsOf,
  targetOf,
} from '../../state/model'
import { nodeCells, loadByRank } from '../../state/nodes'
import { dispatchLine } from '../tasks/Stage'
import { Fn, Legend, Pill, TableScroll, useMeasure } from '../../ui/primitives'
import { Hive, fillRoom } from '../../ui/comb'
import { PageHead, Tabs } from '../../ui/head'
import { Icon } from '../../ui/icons'
import { EvBox, LogBox } from '../../ui/lines'
import { openPorts, openRun, openScale, openConfirm } from '../../sheets'
import { api } from '../../api/client'
import { usePorts } from '../../sheets/port-state'
import { Metrics } from './Metrics'
import { Booting } from './Progress'

const NONE: never[] = []

/** How many of the slowest nodes are named beside the hive, and the fewest a compute must have for naming any to mean something. */
const STRAGGLERS = 5
const ENOUGH = 6

/**
 * The box the drawing fills: as tall as the figures standing beside it, as wide as the rings need.
 *
 * The home is where one compute's size is compared with another's; here the drawing is the way into a
 * node, so it is drawn as large as the card allows — one node is one hexagon the width of its column,
 * not a cell adrift in a column meant for a hundred. Taking its height from the figures is what keeps
 * the two halves ending together, whether the card has nine numbers to show or two.
 */
const ROOM = 400
const TALL: readonly [number, number] = [160, 300]

type Tab = 'logs' | 'events' | 'tasks' | 'spec'

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
  const navigate = useNavigate()
  const c = useStore((s) => computeById(s, id))
  const live = useStore((s) => isLive(s, id))
  const nodes = useStore((s) => s.nodes[id]) ?? NONE
  const tasks = useStore((s) => s.tasks[id]) ?? NONE
  const readings = useStore((s) => s.readings)
  const progress = useStore((s) => s.progress)
  const kin = useStore((s) => [...s.computes, ...s.history].filter((k) => c?.name && k.name === c.name && k.id !== c.id).length)
  const events = useEvents(id)
  const [tab, setTab] = useState<Tab>('logs')
  const [figures, beside] = useMeasure<HTMLDivElement>()
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
  const bound = boundOf(c)
  const size = sizeOf(c, nodes, live)
  const ready = readyOf(nodes)
  const slots = slotsOf(c)
  /*
   * The least busy cells are outlined in the drawing and named beside it: what Stragglers was a
   * list of is pointed at on the map instead, which is only worth doing where there are cells to compare.
   */
  const lowest = live && ready.length >= ENOUGH ? loadByRank(c, nodes, readings).sort((a, b) => a.load - b.load).slice(0, STRAGGLERS) : []
  const cells = nodeCells(c, nodes, readings, progress, new Set(lowest.map((x) => x.rank)))
  const map = fillRoom(cells.length, ROOM, Math.min(TALL[1], Math.max(TALL[0], beside.h || TALL[1])))
  const trouble = c.status.last_error
  const since = events.items.find((e) => e.type === 'compute.degraded' || e.type === 'compute.failed')?.at

  const remove = async () => {
    await api.deleteCompute(c.id)
    navigate('/')
  }

  return (
    <>
      <PageHead
        title={name}
        state={<Pill state={c.status.state} />}
        id={c.name ? c.id : undefined}
        why={trouble ? { message: trouble.message, since: since ? ago(since) : undefined } : null}
        facts={[
          <>
            <b>
              {size.nodes} node{size.nodes === 1 ? '' : 's'}
            </b>
            {size.note ? `, ${size.note}` : ''}
          </>,
          <>
            <b>{machineOf(c)}</b>
            {bound?.instance ? ` ${bound.instance}` : ''}
          </>,
          bound ? (
            <>
              <b>{bound.kind}</b> {bound.region ?? 'any region'}
            </>
          ) : null,
          c.spec.allocation.replace(/_/g, ' '),
          `${c.spec.worker?.executor ?? 'thread'} × ${slots}`,
          live ? `up ${dur(Date.now() - ms(c.created_at))}` : `created ${dateOf(ms(c.created_at))}`,
          live ? null : `ended ${c.ended ? ago(endedAt(c)) : '—'}, ${c.ended ? CAUSE[c.ended.cause] : '—'}`,
          `generation ${c.generation}`,
          live && c.lease.owner ? `held by ${c.lease.owner}` : null,
          kin ? `${kin} other${kin === 1 ? '' : 's'} named ${name}` : null,
        ]}
        primary={live ? { label: 'Run', icon: 'run', onClick: () => openRun({ computeId: c.id }) } : undefined}
        rest={
          live
            ? [
                {
                  label: 'Shell',
                  onClick: () => {
                    useStore.getState().setUi({ shell: true })
                    navigate(`/computes/${c.id}/nodes/${ready[0]?.rank ?? 0}`)
                  },
                },
                { label: 'Scale', onClick: () => openScale(c.id) },
                { label: 'Forward a port', icon: 'ports', onClick: () => openPorts(c.id) },
                {
                  label: 'Delete compute',
                  icon: 'trash',
                  danger: true,
                  onClick: () =>
                    openConfirm({
                      title: `Delete ${name}?`,
                      body: `${nodes.filter((n) => n.state !== 'deleted').length} machines are terminated at the provider.`,
                      confirm: 'Delete',
                      danger: true,
                      onConfirm: () => void remove(),
                    }),
                },
              ]
            : undefined
        }
      />

      <section className="card ov" style={{ '--map': `${map.width}px` }}>
        <div className="map">
          {cells.length ? (
            <Hive
              cells={cells}
              computeId={c.id}
              size={map.size}
              label={`${cells.length} nodes of ${name}`}
              onPick={(rank) => navigate(`/computes/${c.id}/nodes/${rank}`)}
            />
          ) : (
            <span className="sub">No machine was ever attached.</span>
          )}
        </div>
        <div className="groups" ref={figures}>
          {live ? <LiveFigures c={c} nodes={nodes} tasks={tasks} /> : <EndedFigures c={c} nodes={nodes} />}
          <div className="reading">
            <div className="legend">
              <Legend states={nodes.map((n) => n.state)} />
            </div>
            {lowest.length ? (
              <div className="row wrap" style={{ gap: 6 }}>
                <span className="sub" style={{ marginRight: 2 }}>
                  Lowest {acceleratedOf(c) ? 'accelerator' : 'CPU'}
                </span>
                {lowest.map((x) => (
                  <button key={x.rank} className="pill" onClick={() => navigate(`/computes/${c.id}/nodes/${x.rank}`)}>
                    rank {x.rank} <b>{Math.round(x.load)}%</b>
                  </button>
                ))}
              </div>
            ) : null}
            {live && !ready.length ? <Booting nodes={nodes} /> : null}
          </div>
        </div>
      </section>

      <Metrics computeId={c.id} name={name} nodes={nodes} created={ms(c.created_at)} over={live ? undefined : [ms(c.created_at), endedAt(c)]} />

      <section className="card">
        <Tabs<Tab>
          value={tab}
          options={[
            ['logs', 'Logs', null],
            ['events', 'Events', events.items.length || null],
            ['tasks', 'Tasks', callsOf(c) || null],
            ['spec', 'Spec', null],
          ]}
          onChange={setTab}
        >
          <Leaving tab={tab} computeId={c.id} />
        </Tabs>
        <div className="tabbody">
          {tab === 'tasks' ? <TasksTable c={c} tasks={tasks} nodes={nodes} /> : null}
          {tab === 'logs' ? <Logs computeId={c.id} /> : null}
          {tab === 'events' ? <Events computeId={c.id} /> : null}
          {tab === 'spec' ? <SpecGrid c={c} live={live} /> : null}
        </div>
      </section>
    </>
  )
}

/** The one link out of the card, named for where the tab's own list goes on in full. */
function Leaving({ tab, computeId }: { tab: Tab; computeId: string }) {
  const navigate = useNavigate()
  const act = useStore((s) => s.act)
  const task = useStore((s) => s.task)
  const setUi = useStore((s) => s.setUi)
  if (tab === 'spec') return null
  if (tab === 'tasks')
    return (
      <button
        className="btn sm ghost spread"
        onClick={() => {
          setUi({ task: { ...task, compute: computeId, state: 'all' } })
          navigate('/tasks')
        }}
      >
        Open in Tasks
      </button>
    )
  return (
    <button
      className="btn sm ghost spread"
      onClick={() => {
        setUi({ act: { ...act, kind: tab === 'logs' ? 'logs' : 'events', compute: computeId, rank: 'all' } })
        navigate('/activity')
      }}
    >
      Open in Activity
    </button>
  )
}

function Fig({ value, unit, label }: { value: ReactNode; unit?: string; label: string }) {
  return (
    <div className="fig">
      <span className="num">
        {value}
        {unit ? <small>{unit}</small> : null}
      </span>
      <span className="l">{label}</span>
    </div>
  )
}

const Group = ({ label, children }: { label: string; children: ReactNode }) => (
  <div className="group">
    <span className="lbl">{label}</span>
    {children}
  </div>
)

/** What it costs, what it holds and what it is doing — the three questions a running compute is opened for. */
function LiveFigures({ c, nodes, tasks }: { c: Compute; nodes: readonly Node[]; tasks: readonly Task[] }) {
  const readings = useStore((s) => s.readings)
  const functions = useStore((s) => s.functions)
  const spent = useStore((s) => spentOf(s, c)) ?? 0
  const ready = readyOf(nodes)
  const slots = slotsOf(c)
  const busy = ready.reduce((sum, n) => sum + busyOf(tasks, nodes, n.rank), 0)
  const accelerated = acceleratedOf(c)
  const idle = loadByRank(c, nodes, readings).filter((x) => x.load < 25)
  const running = tasks.filter((t) => t.state === 'running')
  const inflight = running.reduce((sum, t) => sum + execsOf(t, nodes).filter((e) => e.state === 'started').length, 0)
  const named = (t: Task) => t.function.name ?? functions[t.function.sha256]?.name ?? t.function.sha256.slice(0, 8)
  const latest = (states: readonly Task['state'][]) => tasks.filter((t) => states.includes(t.state)).sort((a, b) => ms(b.finished_at) - ms(a.finished_at))[0]
  const done = latest(['succeeded'])
  const failed = failedOf(c)
  const finished = finishedOf(c)
  const paid = perUnit(c, nodes)

  const cost = (
    <Group label="Cost">
      <Fig value={money(rateOf(nodes), 2)} unit="/h" label="burning" />
      <Fig value={spent ? money(spent, spent < 10 ? 2 : 0) : '—'} label={finished && spent ? `spent, ${money(spent / finished, spent / finished < 10 ? 2 : 0)} per finished call` : 'spent so far'} />
      <Fig value={paid.rate ? money(paid.rate) : '—'} unit={paid.unit} label={`paid on ${boundOf(c)?.kind ?? '—'}`} />
    </Group>
  )

  /* Nothing is ready: what a page can say about slots, work and idleness is nine zeros, so it says what it is waiting for instead. */
  if (!ready.length)
    return (
      <>
        {cost}
        <Group label="Nodes">
          <Fig value={<>{ready.length}<small>of {targetOf(c)}</small></>} label="ready" />
          <Fig value={nodes.filter((n) => n.state !== 'deleted' && n.state !== 'failed' && n.state !== 'lost').length} label="on their way" />
          <Fig value={nodes.filter((n) => n.state === 'failed' || n.state === 'lost').length} label="lost so far" />
        </Group>
      </>
    )

  return (
    <>
      {cost}
      <Group label="Nodes">
        <Fig value={<>{ready.length}<small>of {targetOf(c)}</small></>} label={`ready${c.spec.nodes.min && c.spec.nodes.min !== c.spec.nodes.initial ? `, floor ${c.spec.nodes.min}` : ''}`} />
        <Fig value={<>{busy}<small>of {ready.length * slots}</small></>} label={`slots busy, ${c.spec.worker?.executor ?? 'thread'} × ${slots}`} />
        <Fig value={idle.length} label={`idle, under 25% ${accelerated ? 'accelerator' : 'CPU'}`} />
      </Group>
      <Group label="Work">
        <Fig value={inflight} label={running.length ? `in flight, ${[...new Set(running.map(named))].join(', ')}` : 'in flight'} />
        <Fig value={c.tasks.succeeded} label={done ? `done, last ${ago(ms(done.finished_at))}` : 'done'} />
        <Fig value={failed} label="errors" />
      </Group>
    </>
  )
}

/** What it cost and what it did, once there is nothing left to watch. */
function EndedFigures({ c, nodes }: { c: Compute; nodes: readonly Node[] }) {
  const spent = c.ended?.cost ?? 0
  const ran = ranOf(c)
  const calls = callsOf(c)
  const failed = failedOf(c)
  const paid = perUnit(c, nodes)
  return (
    <>
      <Group label="Cost">
        <Fig value={spent ? money(spent, spent < 10 ? 2 : 0) : '—'} label="spent in total" />
        <Fig value={calls && spent ? money(spent / calls, spent / calls < 10 ? 2 : 0) : '—'} label="per call" />
        <Fig value={paid.rate ? money(paid.rate) : '—'} unit={paid.unit} label={`paid on ${boundOf(c)?.kind ?? '—'}`} />
      </Group>
      <Group label="Work">
        <Fig value={calls || '—'} label={calls && ran > 60e3 ? `calls, ${Math.round(calls / (ran / HOUR))} an hour` : 'calls'} />
        <Fig value={failed} label={failed && calls ? `failed, ${Math.round((failed / calls) * 1000) / 10}% of them` : 'failed'} />
        <Fig value={ran < 60e3 ? '—' : dur(ran)} label="ran" />
      </Group>
    </>
  )
}

/* ---------- the tabs ---------- */

const attemptOf = (t: Task): number => t.executions.reduce((n, e) => Math.max(n, e.ordinal), 1)

function TasksTable({ c, tasks, nodes }: { c: Compute; tasks: readonly Task[]; nodes: readonly Node[] }) {
  const navigate = useNavigate()
  /* ten of them, in the order the queue reads: what is running, then what is waiting, then the latest to finish */
  const list = tasks.slice().sort(byState).slice(0, 10)
  if (!list.length) return <div className="sub">{callsOf(c) ? 'Its tasks are no longer kept.' : 'Nothing ran here.'}</div>
  return (
    <TableScroll label="Tasks">
      <table>
        <thead>
          <tr>
            <th>Function</th>
            <th>State</th>
            <th>Dispatch</th>
            <th>Submitted</th>
            <th className="right">Took</th>
          </tr>
        </thead>
        <tbody>
          {list.map((t) => {
            const attempt = attemptOf(t)
            const ran = runOf(t)
            const error = t.executions.find((e) => e.error)?.error?.message
            return (
              <tr key={t.id} data-open="" onClick={() => navigate(`/tasks/${t.id}`)}>
                <td>
                  <Fn sha={t.function.sha256} />
                </td>
                <td>
                  <Pill state={t.state} />
                  {error && (t.state === 'failed' || t.state === 'timed_out') ? <span className="err-line">{error}</span> : null}
                </td>
                <td className="sub nowrap">
                  {dispatchLine(t, nodes)}
                  {attempt > 1 ? `, attempt ${attempt}` : ''}
                </td>
                <td className="sub nowrap">{ago(ms(t.submitted_at))}</td>
                <td className="right nowrap">{ran === null ? <span className="faint">—</span> : dur(ran)}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </TableScroll>
  )
}

/** What the compute printed, following the tail as it arrives. */
function Logs({ computeId }: { computeId: string }) {
  const feed = useLogs({ compute: computeId })
  const pageLogs = useStore((s) => s.pageLogs)
  return (
    <LogBox
      lines={feed?.lines ?? NONE}
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

function Events({ computeId }: { computeId: string }) {
  const events = useEvents(computeId)
  const pageEvents = useStore((s) => s.pageEvents)
  return (
    <EvBox
      events={events.items}
      older={
        events.cursor ? (
          <button className="btn sm" style={{ marginBottom: 10 }} disabled={events.loading} onClick={() => void pageEvents(computeId, 'all')}>
            Load older
          </button>
        ) : null
      }
    />
  )
}

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
  const ports = usePorts((s) => s.ports[c.id]) ?? NONE
  const im = c.spec.image
  const pip = im?.pip ?? NONE
  const apt = im?.apt ?? NONE
  const lease = Math.max(0, Math.round((ms(c.lease.expires_at) - Date.now()) / 1000))
  return (
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
            <button className="btn sm ghost" style={{ height: 20, padding: '0 5px', fontSize: 11, marginLeft: 4 }} onClick={() => openPorts(c.id)}>
              <Icon name="ports" />
              Forward
            </button>
          </SpecItem>
        </>
      ) : null}
    </div>
  )
}
