import { memo, useMemo } from 'react'
import { useStore } from '../state/store'
import type { LogLine } from '../state/store'
import type { Compute } from '../api/client'
import { clock } from '../state/model'

export type DockLine = LogLine & { cname?: string; cid?: string }

type Logs = Record<string, LogLine[]>

const nameOf = (computes: readonly Compute[], id: string): string => computes.find((c) => c.id === id)?.name ?? id

/** Every compute's tail, interleaved in the order the lines were printed. */
export const allLogs = (logs: Logs, computes: readonly Compute[]): DockLine[] =>
  Object.entries(logs)
    .flatMap(([cid, lines]) => lines.slice(-120).map<DockLine>((l) => ({ ...l, cname: nameOf(computes, cid), cid })))
    .sort((a, b) => a.at - b.at)
    .slice(-400)

export const scopedLogs = (logs: Logs, computes: readonly Compute[], logRank: 'all' | number, computeId: string | null): DockLine[] =>
  computeId
    ? (logs[computeId] ?? [])
        .filter((l) => logRank === 'all' || l.rank === Number(logRank))
        .slice(-300)
        .map<DockLine>((l) => ({ ...l, cname: nameOf(computes, computeId), cid: computeId }))
    : allLogs(logs, computes)

const Line = memo(function Line({ l }: { l: DockLine }) {
  return (
    <div className={`logline ${l.level === 'err' ? 'err' : l.level === 'warn' ? 'warn' : ''}`}>
      <span>{clock(l.at)}</span>
      <span className="rk">
        <span className="trunc">{l.cname ?? l.cid ?? '—'}</span>
        <span className="rnk">/{l.rank}</span>
      </span>
      <div>{l.text}</div>
    </div>
  )
}, (a, b) => a.l.seq === b.l.seq && a.l.cname === b.l.cname)

export function Logs({ computeId }: { computeId: string | null }) {
  const logs = useStore((s) => s.logs)
  const logRank = useStore((s) => s.logRank)
  const live = useStore((s) => s.computes)
  const history = useStore((s) => s.history)
  const computes = useMemo(() => [...live, ...history], [live, history])
  const list = useMemo(() => scopedLogs(logs, computes, logRank, computeId), [logs, computes, logRank, computeId])
  if (!list.length) return <div className="sub">Nothing printed yet. Lines land here the moment a task writes to stdout.</div>
  return (
    <>
      {list.map((l) => (
        <Line key={`${l.cid ?? ''}-${l.seq}`} l={l} />
      ))}
    </>
  )
}
