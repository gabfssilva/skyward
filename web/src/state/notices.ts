import { useCallback, useMemo, useState } from 'react'
import { api } from '../api/client'
import { recorded, type SkyEvent } from '../api/events'
import { DAY } from './model'
import { usePolled } from './polled'
import { useStore } from './store'

/** The frame names of the events that are a failure. */
export const FAILURES: readonly string[] = ['compute.degraded', 'compute.deletion_failed', 'node.lost', 'node.failed', 'task.failed', 'task.timed_out']

const SAMPLE = 200
const KEY = 'sky-seen'

/** The failures of one kind on one compute for one reason: `ids` newest first, `subject` the newest one's task or node. */
export type Notice = { compute: string; kind: string; reason: string | null; ids: string[]; at: number; subject: string | null }

type Seen = Readonly<Record<string, number>>

/** A failure's kind as a person says it: `node lost`, `task timed out`, `degraded`. */
export const described = (kind: string): string => kind.replace(/^compute\./, '').replace(/[._]/g, ' ')

const reasonOf = (e: SkyEvent): string | null => ('error' in e.data && typeof e.data.error === 'string' ? e.data.error : null)

/** The failures of the last day not yet seen, folded by compute, kind and reason, the group with the newest first. */
export function unseen(events: readonly SkyEvent[], seen: Seen, now: number): Notice[] {
  const failed = new Map(
    events.flatMap((e) => (e.compute && FAILURES.includes(e.frame) && e.at >= now - DAY && !(e.id in seen) ? [[e.id, { ...e, compute: e.compute }] as const] : [])),
  )
  const groups = new Map<string, Notice>()
  for (const e of [...failed.values()].sort((a, b) => b.at - a.at)) {
    const reason = reasonOf(e)
    const key = JSON.stringify([e.compute, e.frame, reason])
    const group = groups.get(key)
    if (group) group.ids.push(e.id)
    else groups.set(key, { compute: e.compute, kind: e.frame, reason, ids: [e.id], at: e.at, subject: e.task ?? e.node ?? null })
  }
  return [...groups.values()]
}

function stored(): Seen {
  const raw: unknown = JSON.parse(localStorage.getItem(KEY) ?? '{}')
  return typeof raw === 'object' && raw !== null ? Object.fromEntries(Object.entries(raw).filter((pair): pair is [string, number] => typeof pair[1] === 'number')) : {}
}

/**
 * What went wrong lately and is not yet seen here, and how to mark some of it seen.
 *
 * A notice is a failure the daemon already wrote in its event log, so the daemon keeps nothing new for it: the
 * last day of failures is read off the log, and what the stream says since is added as it arrives. What has been
 * seen belongs to the reader and stays in this browser, each event id with when it was marked; what was marked a
 * day ago failed earlier still, and is forgotten.
 */
export function useNotices(): { notices: Notice[]; see: (ids: readonly string[]) => void } {
  const read = usePolled(async () => (await api.log({ types: FAILURES, limit: SAMPLE })).items.map(recorded), 'notices', [] as SkyEvent[])
  const streamed = useStore((s) => s.events)
  const [seen, setSeen] = useState(stored)
  const notices = useMemo(() => unseen([...streamed, ...read], seen, Date.now()), [streamed, read, seen])
  const see = useCallback(
    (ids: readonly string[]) => {
      const now = Date.now()
      const kept = Object.fromEntries(Object.entries(seen).filter(([, at]) => at >= now - DAY))
      const next = { ...kept, ...Object.fromEntries(ids.map((id) => [id, now])) }
      localStorage.setItem(KEY, JSON.stringify(next))
      setSeen(next)
    },
    [seen],
  )
  return { notices, see }
}
