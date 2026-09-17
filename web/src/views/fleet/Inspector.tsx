import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../../api/client'
import type { Compute, Task } from '../../api/client'
import { recorded, type SkyEvent } from '../../api/events'
import { useStore } from '../../state/store'
import { HOUR, METRICS, ago, dur, money, ms, rateOf, type MetricKey } from '../../state/model'
import { Histo } from '../../ui/charts'
import { Pick, Tick } from '../../ui/primitives'
import { valuesFor } from '../../state/nodes'

const HUES = ['--c1', '--c2', '--c3', '--c4', '--c5'] as const

const EVERY = 30_000
const SAMPLE = 50
const LOSSES: readonly string[] = ['node.lost', 'node.failed']

export function Inspector() {
  const navigate = useNavigate()
  const computes = useStore((s) => s.computes)
  const nodesByCompute = useStore((s) => s.nodes)
  const metrics = useStore((s) => s.metrics)
  const costs = useStore((s) => s.costs)
  const metric = useStore((s) => s.metric)
  const setUi = useStore((s) => s.setUi)

  const readyCount = computes.reduce((n, c) => n + (nodesByCompute[c.id] ?? []).filter((x) => x.state === 'ready').length, 0)
  const across = computes.flatMap((c) => valuesFor(c.id, nodesByCompute[c.id] ?? [], metrics, metric))

  const split = computes
    .map((c, i) => ({ c, spent: costs[c.id], hue: HUES[i % 5]! }))
    .sort((a, b) => (b.spent ?? 0) - (a.spent ?? 0))
  const total = split.reduce((sum, x) => sum + (x.spent ?? 0), 0)
  const metered = split.some((x) => x.spent !== undefined)

  return (
    <>
      <Attention />

      <section className="card tight">
        <div className="cap">Spent so far</div>
        <div className="gauge-r" style={{ margin: '8px 0 10px' }}>
          <b>{metered ? money(total, total < 10 ? 2 : 0) : '—'}</b>
          <span>since each running compute started</span>
        </div>
        <div style={{ display: 'flex', gap: 3 }}>
          {split.map((x) => (
            <div
              key={x.c.id}
              data-tip={`${x.c.name} · ${x.spent === undefined ? 'not metered yet' : money(x.spent)}`}
              style={{ flex: Math.max(x.spent ?? 0, 0.01), height: 8, borderRadius: 3, background: `var(${x.hue})` }}
            />
          ))}
        </div>
        <div style={{ marginTop: 8 }}>
          {split.map((x) => (
            <div className="kv" key={x.c.id}>
              <button className="row" style={{ gap: 6 }} onClick={() => navigate(`/computes/${x.c.id}`)}>
                <i style={{ width: 9, height: 9, borderRadius: 3, background: `var(${x.hue})` }} />
                {x.c.name}
              </button>
              <span className="mono">{x.spent === undefined ? '—' : money(x.spent)}</span>
            </div>
          ))}
        </div>
      </section>

      <Queue />

      <section className="card tight">
        <div className="cap">Across {readyCount} nodes</div>
        <div style={{ marginTop: 8 }}>
          <Pick<MetricKey> value={metric} options={METRICS} onChange={(v) => setUi({ metric: v })} />
        </div>
        <div style={{ marginTop: 9 }}>
          <Histo values={across} metric={metric} />
        </div>
      </section>
    </>
  )
}

/**
 * What is left to run across the fleet, when it will be done, and what it costs to get there.
 *
 * The pace is read from the daemon — each working compute's latest finished tasks — rather than
 * timed in this page, so a console opened in the middle of a run knows it at once.
 */
function Queue() {
  const navigate = useNavigate()
  const computes = useStore((s) => s.computes)
  const nodesByCompute = useStore((s) => s.nodes)

  const working = computes.filter((c) => c.tasks.queued + c.tasks.running > 0)
  const ids = working.map((c) => c.id)
  const finished = usePolled(
    async () => ({
      at: Date.now(),
      tasks: Object.fromEntries(await Promise.all(ids.map(async (id) => [id, (await api.tasks({ compute: id, order: 'finished', limit: SAMPLE })).items] as const))),
    }),
    ids.join(','),
    { at: Date.now(), tasks: {} as Record<string, Task[]> },
  )

  const rows = working.map((c) => {
    const left = c.tasks.queued + c.tasks.running
    const pace = paceOf(finished.tasks[c.id] ?? [], finished.at)
    const eta = pace === null ? null : left / pace
    return { c, left, pace, eta, cost: eta === null ? null : (rateOf(nodesByCompute[c.id] ?? []) * eta) / HOUR }
  })
  const paced = rows.every((r) => r.eta !== null)
  const eta = Math.max(0, ...rows.map((r) => r.eta ?? 0))
  const cost = rows.reduce((sum, r) => sum + (r.cost ?? 0), 0)

  const count = (of: (c: Compute) => number) => computes.reduce((n, c) => n + of(c), 0)
  const parts = [
    { label: 'succeeded', n: count((c) => c.tasks.succeeded), hue: '--ok' },
    { label: 'failed', n: count((c) => c.tasks.failed + c.tasks.timed_out), hue: '--bad' },
    { label: 'running', n: count((c) => c.tasks.running), hue: '--boot' },
    { label: 'queued', n: count((c) => c.tasks.queued), hue: '--edge' },
  ]

  return (
    <section className="card tight">
      <div className="cap">Queue</div>
      {rows.length ? (
        <div className="gauge-r" style={{ margin: '8px 0 10px' }}>
          <b>{paced ? dur(eta) : '—'}</b>
          <span>{paced ? `to finish · ~${money(cost)} more` : 'no pace yet — fewer than 3 tasks finished in the last hour'}</span>
        </div>
      ) : (
        <div className="sub" style={{ margin: '8px 0 10px' }}>
          nothing queued or running
        </div>
      )}
      {parts.some((x) => x.n) ? (
        <>
          <div style={{ display: 'flex', gap: 3 }}>
            {parts
              .filter((x) => x.n)
              .map((x) => (
                <div key={x.label} data-tip={`${x.n} ${x.label}`} style={{ flex: x.n, height: 8, borderRadius: 3, background: `var(${x.hue})` }} />
              ))}
          </div>
          <div className="faint mono" style={{ fontSize: 11, marginTop: 6 }}>
            {parts.map((x) => `${x.n} ${x.label}`).join(' · ')}
          </div>
        </>
      ) : null}
      <div style={{ marginTop: 8 }}>
        {rows.map((r) => (
          <div className="kv" key={r.c.id}>
            <button style={{ flex: 1, minWidth: 0 }} onClick={() => navigate(`/computes/${r.c.id}`)}>
              <Tick text={r.c.name ?? r.c.id} />
            </button>
            <span className="mono" style={{ whiteSpace: 'nowrap' }} data-tip={r.pace === null ? 'no pace yet' : `${(r.pace * 60_000).toFixed(1)} tasks a minute`}>
              {r.left} left · {r.eta === null ? '—' : dur(r.eta)}
              {r.cost === null ? null : <span className="faint"> · {money(r.cost)}</span>}
            </span>
          </div>
        ))}
      </div>
    </section>
  )
}

type Trouble = { key: string; tone: '--bad' | '--warn'; title: string; what: string; detail?: string; open: () => void }

/**
 * What in the fleet is wrong right now, and nothing when nothing is.
 *
 * A lost machine leaves the node listing, so losses are read off the daemon's log, the last hour
 * of them; everything else is a state the store already holds.
 */
function Attention() {
  const navigate = useNavigate()
  const computes = useStore((s) => s.computes)
  const providers = useStore((s) => s.providers)
  const setUi = useStore((s) => s.setUi)
  const losses = usePolled(
    async () => ({ at: Date.now(), events: (await api.log({ types: LOSSES, limit: SAMPLE })).items.map(recorded) }),
    '',
    { at: Date.now(), events: [] as SkyEvent[] },
  )

  const troubles: Trouble[] = [
    ...computes
      .filter((c) => c.status.state === 'degraded')
      .map((c): Trouble => ({ key: `degraded/${c.id}`, tone: '--bad', title: c.name ?? c.id, what: 'degraded', detail: c.status.last_error?.message, open: () => navigate(`/computes/${c.id}`) })),
    ...computes.flatMap((c): Trouble[] => {
      const lost = losses.events.filter((e) => e.compute === c.id && e.at > losses.at - HOUR)
      const last = lost[0]
      if (!last) return []
      return [
        {
          key: `nodes/${c.id}`,
          tone: '--bad',
          title: c.name ?? c.id,
          what: `${lost.length} node${lost.length === 1 ? '' : 's'} lost in the last hour · latest ${ago(last.at)}`,
          detail: last.data.type === 'node.state' ? (last.data.error ?? undefined) : undefined,
          open: () => navigate(`/computes/${c.id}`),
        },
      ]
    }),
    ...computes
      .filter((c) => c.tasks.failed + c.tasks.timed_out > 0)
      .map(
        (c): Trouble => ({
          key: `tasks/${c.id}`,
          tone: '--bad',
          title: c.name ?? c.id,
          what: [c.tasks.failed ? `${c.tasks.failed} task${c.tasks.failed === 1 ? '' : 's'} failed` : '', c.tasks.timed_out ? `${c.tasks.timed_out} timed out` : '']
            .filter(Boolean)
            .join(' · '),
          open: () => {
            setUi({ task: { compute: c.id, state: c.tasks.failed ? 'failed' : 'timed_out' } })
            navigate('/tasks')
          },
        }),
      ),
    ...providers
      .filter((p) => p.last_error)
      .map((p): Trouble => ({ key: `provider/${p.id}`, tone: '--warn', title: p.name, what: 'could not refresh its offers', detail: p.last_error?.message, open: () => navigate('/providers') })),
  ]

  if (!troubles.length) return null

  return (
    <section className="card tight">
      <div className="cap">Needs attention</div>
      <div style={{ marginTop: 8 }}>
        {troubles.map((t) => (
          <button key={t.key} className="trouble" onClick={t.open}>
            <i style={{ background: `var(${t.tone})` }} />
            <span>
              <Tick text={t.title} />
              <span className="sub">{t.what}</span>
              {t.detail ? (
                <span className="faint trunc" data-tip={t.detail}>
                  {t.detail}
                </span>
              ) : null}
            </span>
          </button>
        ))}
      </div>
    </section>
  )
}

/**
 * How many tasks a compute finishes a millisecond: the ones that ended in the last hour, over the time since the oldest
 * of them. It runs up to now rather than to the latest finish, so a compute that stops finishing slows down here too;
 * fewer than three is no pace at all.
 */
const paceOf = (finished: readonly Task[], now: number): number | null => {
  const ends = finished.map((t) => ms(t.finished_at)).filter((at) => at > now - HOUR)
  return ends.length < 3 ? null : ends.length / (now - Math.min(...ends))
}

/** What ``read`` answers now and every half minute after, while the card is up and ``key`` stays the same; a read that fails keeps the last answer. */
function usePolled<T>(read: () => Promise<T>, key: string, initial: T): T {
  const [value, setValue] = useState(initial)
  useEffect(() => {
    let live = true
    const once = () =>
      read().then(
        (answer) => live && setValue(answer),
        () => undefined,
      )
    void once()
    const every = setInterval(once, EVERY)
    return () => {
      live = false
      clearInterval(every)
    }
  }, [key])
  return value
}
