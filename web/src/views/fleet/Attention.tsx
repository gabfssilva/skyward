import { useNavigate } from 'react-router-dom'
import { api } from '../../api/client'
import { recorded, type SkyEvent } from '../../api/events'
import { HOUR, ago, failedOf, ms } from '../../state/model'
import { usePolled } from '../../state/polled'
import { useStore } from '../../state/store'
import { Icon } from '../../ui/icons'

const SAMPLE = 50
const LOSSES: readonly string[] = ['node.lost', 'node.failed']

type Trouble = { key: string; sev: 'bad' | 'warn'; who: string; what: string; detail?: string; open: () => void }

/** One line of trouble: what it is about, what is wrong, and the detail that says what to do about it. */
function Row({ t }: { t: Trouble }) {
  return (
    <button className="arow" onClick={t.open}>
      <i className={`sev ${t.sev}`} />
      <span className="trunc">
        <b>{t.who}</b> <span className="what">{t.what}</span>
      </span>
      <span className="detail">{t.detail ?? ''}</span>
      <Icon name="next" />
    </button>
  )
}

/**
 * What in the fleet is wrong right now, first on the page, and nothing at all when nothing is.
 *
 * A lost machine leaves the node listing, so losses are read off the daemon's log, the last hour of
 * them; everything else is a state the store already holds. Each line says what to open, because a
 * warning with nowhere to go is just a colour.
 */
export function Attention() {
  const navigate = useNavigate()
  const computes = useStore((s) => s.computes)
  const tasks = useStore((s) => s.tasks)
  const providers = useStore((s) => s.providers)
  const functions = useStore((s) => s.functions)
  const setUi = useStore((s) => s.setUi)
  const losses = usePolled(async () => ({ at: Date.now(), events: (await api.log({ types: LOSSES, limit: SAMPLE })).items.map(recorded) }), '', {
    at: Date.now(),
    events: [] as SkyEvent[],
  })

  const troubles: Trouble[] = [
    ...computes
      .filter((c) => c.status.state === 'degraded')
      .map((c): Trouble => ({
        key: `degraded/${c.id}`,
        sev: 'bad',
        who: c.name ?? c.id,
        what: 'degraded',
        detail: c.status.last_error?.message,
        open: () => navigate(`/computes/${c.id}`),
      })),
    ...computes.flatMap((c): Trouble[] => {
      const lost = losses.events.filter((e) => e.compute === c.id && e.at > losses.at - HOUR)
      const last = lost[0]
      if (!last) return []
      return [
        {
          key: `nodes/${c.id}`,
          sev: 'bad',
          who: c.name ?? c.id,
          what: `${lost.length} node${lost.length === 1 ? '' : 's'} lost in the last hour`,
          detail: `${last.data.type === 'node.state' ? `${last.data.error ?? last.data.state}, ` : ''}latest ${ago(last.at)}`,
          open: () => navigate(`/computes/${c.id}`),
        },
      ]
    }),
    ...computes
      .filter((c) => failedOf(c) > 0)
      .map((c): Trouble => {
        const latest = (tasks[c.id] ?? []).filter((t) => t.state === 'failed' || t.state === 'timed_out').sort((a, b) => ms(b.finished_at) - ms(a.finished_at))[0]
        const named = latest && (latest.function.name ?? functions[latest.function.sha256]?.name ?? latest.function.sha256.slice(0, 8))
        return {
          key: `tasks/${c.id}`,
          sev: 'bad',
          who: c.name ?? c.id,
          what: `${failedOf(c)} task${failedOf(c) === 1 ? '' : 's'} failed`,
          detail: latest ? `${named}, ${ago(ms(latest.finished_at))}` : undefined,
          open: () => {
            setUi({ task: { compute: c.id, state: 'failed', lineage: 'all' } })
            navigate('/tasks')
          },
        }
      }),
    ...providers
      .filter((p) => p.last_error)
      .map((p): Trouble => ({
        key: `provider/${p.id}`,
        sev: 'warn',
        who: `${p.kind}${p.name && p.name !== 'default' ? ` ${p.name}` : ''}`,
        what: 'offers not refreshed',
        detail: p.last_error?.message,
        open: () => navigate('/market/accounts'),
      })),
  ]

  if (!troubles.length) return null

  return (
    <section className="card att">
      <div className="chead" style={{ marginBottom: 2 }}>
        <span className="h">Needs attention</span>
        <span className="sub">{troubles.length}</span>
      </div>
      {troubles.map((t) => (
        <Row key={t.key} t={t} />
      ))}
    </section>
  )
}
