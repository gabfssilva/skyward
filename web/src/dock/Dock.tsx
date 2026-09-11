import { useEffect, useMemo, useRef } from 'react'
import { useConsoles, useStore } from '../state/store'
import type { DockTab } from '../state/store'
import { Icon } from '../ui/icons'
import { clock, readyOf } from '../state/model'
import { useDockScope } from './scope'
import { Logs, allLogs } from './Logs'
import { Events } from './Events'
import { Tasks } from './Tasks'
import { Shell } from './Shell'

const DOCKS: readonly (readonly [DockTab, string])[] = [
  ['logs', 'Logs'],
  ['events', 'Events'],
  ['tasks', 'Tasks'],
  ['shell', 'Shell'],
]

export function Dock() {
  const computeId = useDockScope()
  useConsoles(computeId)
  const dock = useStore((s) => s.dock)
  const dockMin = useStore((s) => s.dockMin)
  const logRank = useStore((s) => s.logRank)
  const logFollow = useStore((s) => s.logFollow)
  const setUi = useStore((s) => s.setUi)
  const computes = useStore((s) => s.computes)
  const history = useStore((s) => s.history)
  const logs = useStore((s) => s.logs)
  const allNodes = useStore((s) => s.nodes)
  const body = useRef<HTMLDivElement>(null)

  const compute = computes.find((c) => c.id === computeId) ?? history.find((c) => c.id === computeId)
  const lines = useMemo(
    () => (compute ? (logs[compute.id] ?? []) : allLogs(logs, computes)),
    [compute, logs, computes],
  )
  const tail = lines[lines.length - 1]
  const ranks = compute ? readyOf(allNodes[compute.id] ?? []).slice(0, 4).map((n) => n.rank) : []

  useEffect(() => {
    const el = body.current
    if (el && dock === 'logs' && logFollow) el.scrollTop = el.scrollHeight
  })

  return (
    <div className={dockMin ? 'dock min' : 'dock'} id="dock">
      <div className="dock-bar" id="dock-bar">
        <div className="pick">
          {DOCKS.map(([k, label]) => (
            <button key={k} aria-selected={dock === k} onClick={() => setUi({ dock: k })}>
              <Icon name={k} />
              {label}
            </button>
          ))}
        </div>
        {dockMin ? (
          <span className="tail">{tail ? `${clock(tail.at)}  ${tail.text}` : ''}</span>
        ) : (
          <span className="tail">{compute ? (compute.name ?? compute.id) : 'every compute'}</span>
        )}
        {dock === 'logs' && compute ? (
          <div className="row" style={{ gap: 6 }}>
            <span className="cap">rank</span>
            <div className="pick">
              {(['all', ...ranks] as (string | number)[]).map((r) => (
                <button
                  key={r}
                  aria-selected={String(logRank) === String(r)}
                  onClick={() => setUi({ logRank: r === 'all' ? 'all' : Number(r) })}
                >
                  {r}
                </button>
              ))}
            </div>
          </div>
        ) : null}
        {dock === 'logs' ? (
          <button className="chip" aria-pressed={logFollow} onClick={() => setUi({ logFollow: !logFollow })}>
            {logFollow ? 'following' : 'paused'}
          </button>
        ) : null}
        <button className="btn sm ghost" onClick={() => setUi({ dockMin: !dockMin })}>
          <Icon name={dockMin ? 'hide' : 'show'} />
          {dockMin ? 'Open' : 'Hide'}
        </button>
      </div>
      <div className="dock-body" id="dock-body" ref={body}>
        {dockMin ? null : dock === 'logs' ? (
          <Logs computeId={computeId} />
        ) : dock === 'events' ? (
          <Events computeId={computeId} />
        ) : dock === 'tasks' ? (
          <Tasks computeId={computeId} />
        ) : (
          <Shell computeId={computeId} />
        )}
      </div>
    </div>
  )
}

export default Dock
