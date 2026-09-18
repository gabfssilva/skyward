import { useEffect, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../../api/client'
import type { FunctionRef, Node, Task } from '../../api/client'
import { computeById, freshest, useLibrary, useStore, useTasks } from '../../state/store'
import type { TaskFilters } from '../../state/store'
import { useLineage, versionsOf, whereOf } from '../../state/functions'
import { usePolled } from '../../state/polled'
import { DAY, HOUR, ago, dur, execsOf, ms, runOf } from '../../state/model'
import { Fn, Pill, TableScroll, useMeasure } from '../../ui/primitives'
import { PageHead, Actions, Facts } from '../../ui/head'
import { Icon } from '../../ui/icons'
import { openFunction, openRun, openWrite } from '../../sheets'

const NONE: never[] = []

/** How much of the queue one read carries: every running and queued task, then the latest to finish. */
const WATCHED = 100

/** The width the card needs to hold the function list beside the tasks. */
const ROOM = 760

const attemptOf = (t: Task): number => t.executions.reduce((n, e) => Math.max(n, e.ordinal), 1)

/**
 * How many ranks a task reached: the executions it is drawn with, which for a broadcast are filled in
 * for the ranks the daemon has not recorded one on yet.
 */
export const ranksOf = (t: Task, nodes: readonly Node[] = []): number => new Set(execsOf(t, nodes).map((e) => e.rank)).size

/** Where a task went. A broadcast says how many ranks, unless nothing is left to count them on. */
export const dispatchLine = (t: Task, nodes: readonly Node[] = []): string => {
  if (t.dispatch === 'one') return 'one node'
  const ranks = ranksOf(t, nodes)
  const what = t.dispatch === 'all' ? 'every node' : 'stream'
  return ranks ? `${what}, ${ranks}` : what
}

/** The states worth a chip of their own; the rest are in the menu the last one opens. */
const SHOWN: readonly (readonly [TaskFilters['state'], string])[] = [
  ['all', 'All'],
  ['running', 'Running'],
  ['failed', 'Failed'],
  ['succeeded', 'Succeeded'],
]

const REST: readonly (readonly [TaskFilters['state'], string])[] = [
  ['queued', 'Queued'],
  ['cancelled', 'Cancelled'],
  ['timed_out', 'Timed out'],
  ['indeterminate', 'Indeterminate'],
]

const stateOf = (value: string): TaskFilters['state'] => [...SHOWN, ...REST].find(([state]) => state === value)?.[0] ?? 'all'

/**
 * How many tasks a compute finishes a millisecond: the ones that ended in the last hour, over the time since the
 * oldest of them. It runs up to now rather than to the latest finish, so a fleet that stops finishing slows down
 * here too; fewer than three is no pace at all.
 */
const paceOf = (finished: readonly Task[], now: number): number | null => {
  const ends = finished.map((t) => ms(t.finished_at)).filter((at) => at > now - HOUR)
  return ends.length < 3 ? null : ends.length / (now - Math.min(...ends))
}

/**
 * What the queue is doing, in one read.
 *
 * ``order: 'state'`` answers with everything running, then everything queued, then the latest to finish — so
 * one page carries the counts, the errors of the last day and the pace, and the page does not ask a question
 * per function.
 */
function useQueue() {
  const learn = useStore((s) => s.learnFunctions)
  const watched = usePolled(async () => ({ at: Date.now(), tasks: (await api.tasks({ order: 'state', limit: WATCHED })).items }), 'queue', {
    at: Date.now(),
    tasks: NONE as readonly Task[],
  })
  useEffect(() => {
    if (watched.tasks.length) void learn(watched.tasks)
  }, [watched, learn])
  return useMemo(() => {
    const running = watched.tasks.filter((t) => t.state === 'running')
    const queued = watched.tasks.filter((t) => t.state === 'queued')
    const finished = watched.tasks.filter((t) => t.finished_at !== null)
    const failed = finished.filter((t) => (t.state === 'failed' || t.state === 'timed_out') && ms(t.finished_at) > watched.at - DAY)
    const pace = paceOf(finished, watched.at)
    return { running, queued, failed, eta: pace === null ? null : (running.length + queued.length) / pace }
  }, [watched])
}

/** One task of the feed: the freshest copy the store has of it, and the error that made it stop. */
function Row({ task, named }: { task: Task; named: boolean }) {
  const navigate = useNavigate()
  const t = useStore((s) => freshest(s, task))
  const c = useStore((s) => computeById(s, task.compute.id))
  const nodes = useStore((s) => s.nodes[task.compute.id]) ?? NONE
  const live = useStore((s) => s.computes.some((x) => x.id === task.compute.id))
  const attempt = attemptOf(t)
  const ran = runOf(t)
  const error = t.executions.find((e) => e.error)?.error?.message
  return (
    <tr data-open="" onClick={() => navigate(`/tasks/${t.id}`)}>
      {named ? (
        <td data-cell="fn">
          <Fn sha={t.function.sha256} />
        </td>
      ) : null}
      <td data-cell={named ? 'state' : 'fn'} style={{ maxWidth: 300 }}>
        <Pill state={t.state} />
        {error && (t.state === 'failed' || t.state === 'timed_out') ? <span className="err-line">{error}</span> : null}
      </td>
      <td data-cell="where" className="nowrap">
        <b style={{ fontWeight: 600 }}>{c?.name ?? task.compute.name ?? task.compute.id}</b>
        {c && !live ? <span className="faint"> ended</span> : null}
      </td>
      <td data-cell="hide" className="sub nowrap">
        {dispatchLine(t, nodes)}
        {attempt > 1 ? `, attempt ${attempt}` : ''}
      </td>
      <td data-cell="hide" className="sub nowrap">
        {ago(ms(t.submitted_at))}
      </td>
      <td data-cell="took" className="right nowrap">
        {ran === null ? <span className="faint">—</span> : dur(ran)}
      </td>
    </tr>
  )
}

/** The functions the daemon holds, with what each of them is doing right now. */
function Library({ running, failed }: { running: readonly Task[]; failed: readonly Task[] }) {
  const library = useLibrary()
  const functions = useStore((s) => s.functions)
  const f = useStore((s) => s.task)
  const setUi = useStore((s) => s.setUi)
  const rows = library?.items ?? NONE
  const lineageOf = (t: Task): string | undefined => functions[t.function.sha256]?.lineage
  const count = (tasks: readonly Task[], fn: FunctionRef) => tasks.filter((t) => lineageOf(t) === fn.lineage).length

  return (
    <div className="fnlist">
      <button className="fn" aria-selected={f.lineage === 'all'} onClick={() => setUi({ task: { ...f, lineage: 'all' } })}>
        <b>All functions</b>
        <span className="cnts sub">{library?.total ?? rows.length}</span>
      </button>
      {rows.map((fn) => {
        const up = count(running, fn)
        const bad = count(failed, fn)
        return (
          <button key={fn.lineage} className="fn" aria-selected={f.lineage === fn.lineage} onClick={() => setUi({ task: { ...f, lineage: fn.lineage } })}>
            <b>{fn.name ?? fn.sha256.slice(0, 8)}</b>
            <span className="cnts">
              {up ? (
                <span className="cnt running" data-tip={`${up} running`}>
                  <i />
                  {up}
                </span>
              ) : null}
              {bad ? (
                <span className="cnt failed" data-tip={`${bad} failed in the last day`}>
                  <i />
                  {bad}
                </span>
              ) : null}
            </span>
          </button>
        )
      })}
      <button className="fnnew" onClick={() => openWrite()}>
        <Icon name="plus" />
        New function
      </button>
    </div>
  )
}

/** The function a name was picked from, where the list has to be a control instead of a column. */
function Picker() {
  const library = useLibrary()
  const f = useStore((s) => s.task)
  const setUi = useStore((s) => s.setUi)
  const rows = library?.items ?? NONE
  return (
    <span className="picker">
      <select
        className="search"
        aria-label="function"
        style={{ background: 'none', minWidth: 0, flex: 1, height: 28, padding: 0, fontWeight: 600 }}
        value={f.lineage}
        onChange={(e) => setUi({ task: { ...f, lineage: e.target.value } })}
      >
        <option value="all">All functions</option>
        {rows.map((fn) => (
          <option key={fn.lineage} value={fn.lineage}>
            {fn.name ?? fn.sha256.slice(0, 8)}
          </option>
        ))}
      </select>
    </span>
  )
}

/** What one function is, above the tasks of it: its newest version, where it was written, and how to run it. */
function Chosen({ lineage }: { lineage: string }) {
  const { uploads } = useLineage(lineage)
  const feed = useTasks()
  const versions = versionsOf(uploads)
  const newest = versions[0]?.newest
  const total = feed?.total ?? null
  const last = feed?.items[0]
  if (!newest) return null
  return (
    <>
      <div className="row wrap">
        <span className="num">
          {newest.name ?? newest.sha256.slice(0, 8)}
          <span className="v">v{newest.version}</span>
        </span>
        <span className="sub">{whereOf(newest)}</span>
        <Actions
          primary={{ label: 'Run', icon: 'run', onClick: () => openRun({ lineage }) }}
          rest={[
            { label: 'Code', onClick: () => openFunction(lineage) },
            newest.source ? { label: 'Edit', onClick: () => openWrite(newest) } : null,
          ]}
        />
      </div>
      <Facts
        items={[
          total === null ? null : (
            <>
              <b>{total}</b> task{total === 1 ? '' : 's'}
            </>
          ),
          last ? `last run ${ago(ms(last.submitted_at))}` : null,
          `${versions.length} version${versions.length === 1 ? '' : 's'}`,
          newest.source ? 'written here' : 'sent by the SDK',
        ]}
      />
    </>
  )
}

/**
 * Tasks and functions in one page.
 *
 * It opens on the whole queue — what is running, then what is waiting, then the latest to finish — with the
 * functions beside it. Picking one narrows the table to that function and puts what the function is above it;
 * nothing is hidden behind a second page, because a function and the tasks of it are the same question asked twice.
 */
export function Stage() {
  const feed = useTasks()
  const computes = useStore((s) => s.computes)
  const history = useStore((s) => s.history)
  const f = useStore((s) => s.task)
  const setUi = useStore((s) => s.setUi)
  const pageTasks = useStore((s) => s.pageTasks)
  const [card, { w }] = useMeasure<HTMLElement>()
  const beside = w === 0 || w >= ROOM
  const queue = useQueue()
  const set = (patch: Partial<TaskFilters>) => setUi({ task: { ...f, ...patch } })
  const rows = feed?.items ?? NONE
  const named = f.lineage === 'all'
  const listed = [...computes, ...history].some((c) => c.id === f.compute)

  return (
    <>
      <PageHead
        primary={beside ? undefined : { label: 'Run', icon: 'run', onClick: () => openRun(named ? {} : { lineage: f.lineage }) }}
        rest={beside ? undefined : [{ label: 'New function', icon: 'plus', onClick: () => openWrite() }]}
      >
        <div className="sum">
          <span>
            <b>{queue.running.length}</b> running
          </span>
          <span>
            <b>{queue.queued.length}</b> queued
          </span>
          <span>
            <b>{queue.failed.length}</b> failed in 24h
          </span>
          <span>
            <b>{queue.eta === null ? '—' : dur(queue.eta)}</b> to finish
            {queue.eta === null ? ', fewer than 3 tasks ended in the last hour' : ''}
          </span>
        </div>
      </PageHead>

      <section className={beside ? 'card flush tasks' : 'card flush tasks alone'} ref={card}>
        {beside ? <Library running={queue.running} failed={queue.failed} /> : null}
        <div className="feed">
          {named ? null : <Chosen lineage={f.lineage} />}
          <div className="row wrap">
            {beside ? null : <Picker />}
            <div className="pick">
              {SHOWN.map(([state, label]) => (
                <button key={state} aria-selected={f.state === state} onClick={() => set({ state })}>
                  {label}
                </button>
              ))}
            </div>
            <select
              className="search"
              aria-label="more states"
              style={{ minWidth: 104 }}
              value={REST.some(([state]) => state === f.state) ? f.state : ''}
              onChange={(e) => set({ state: stateOf(e.target.value) })}
            >
              <option value="">More…</option>
              {REST.map(([state, label]) => (
                <option key={state} value={state}>
                  {label}
                </option>
              ))}
            </select>
            <select className="search" aria-label="compute" style={{ minWidth: 140, marginLeft: 'auto' }} value={f.compute} onChange={(e) => set({ compute: e.target.value })}>
              <option value="all">Every compute</option>
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
              {f.compute !== 'all' && !listed ? <option value={f.compute}>{f.compute}</option> : null}
            </select>
            {named && beside ? <Actions primary={{ label: 'Run', icon: 'run', onClick: () => openRun() }} /> : null}
          </div>
          <TableScroll label="Tasks">
            <table className="cards tasklist">
              <thead>
                <tr>
                  {named ? <th>Function</th> : null}
                  <th>State</th>
                  <th>Compute</th>
                  <th>Dispatch</th>
                  <th>Submitted</th>
                  <th className="right">Took</th>
                </tr>
              </thead>
              <tbody>
                {rows.length ? (
                  rows.map((t) => <Row key={t.id} task={t} named={named} />)
                ) : (
                  <tr>
                    <td colSpan={6} className="sub" style={{ padding: '22px 0', textAlign: 'center' }}>
                      {(feed?.loading ?? true) ? 'Reading tasks…' : 'No task matches these filters.'}
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </TableScroll>
          {feed?.cursor ? (
            <button className="btn sm ghost" style={{ justifySelf: 'start' }} disabled={feed.loading} onClick={() => void pageTasks()}>
              Show older
            </button>
          ) : null}
        </div>
      </section>
    </>
  )
}
