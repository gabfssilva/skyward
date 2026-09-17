import type { ReactNode } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { api } from '../../api/client'
import type { Compute, Node } from '../../api/client'
import { useStore, computeById, isLive, spentOf } from '../../state/store'
import { valuesFor } from '../../state/nodes'
import { callsOf, dur, median, money, nodeLive, ranOf, rateOf, readyOf, targetOf } from '../../state/model'
import { Icon } from '../../ui/icons'
import { openRun } from '../../sheets'

const NONE: never[] = []

/** One figure of the compact rail: a bold value and what it is. */
export function Stat({ value, label, unit }: { value: ReactNode; label: string; unit?: string }) {
  return (
    <span className="stat">
      <b>
        {value}
        {unit ? <small>{unit}</small> : null}
      </b>
      <span>{label}</span>
    </span>
  )
}

/** The compute's stats, the way the node and task rails carry them. */
export function ComputeStats({ c, nodes, live }: { c: Compute; nodes: readonly Node[]; live: boolean }) {
  const metrics = useStore((s) => s.metrics)
  const spent = useStore((s) => spentOf(s, c))
  if (!live) {
    const cost = c.ended?.cost ?? 0
    const ran = ranOf(c)
    return (
      <>
        <Stat value={cost ? money(cost, cost < 10 ? 2 : 0) : '—'} label="spent" />
        <Stat value={ran < 60e3 ? '—' : dur(ran)} label="ran" />
        <Stat value={nodes.length || targetOf(c)} label="nodes" />
        <Stat value={callsOf(c) || '—'} label="calls" />
      </>
    )
  }
  const gpu = valuesFor(c.id, nodes, metrics, 'gpu')
  return (
    <>
      <Stat value={money(rateOf(nodes), 2)} label="per hour" unit="/h" />
      <Stat
        value={
          <>
            {readyOf(nodes).length}
            <small>/{targetOf(c)}</small>
          </>
        }
        label="ready"
      />
      <Stat value={Math.round(median(gpu) || 0)} label="median GPU" unit="%" />
      <Stat value={spent === undefined ? '—' : money(spent, spent < 10 ? 2 : 0)} label="spent" />
    </>
  )
}

/** The actions on a compute: Logs, Scale, Shell and Delete while it runs, Events once it is gone. */
export function ComputeActions({ c, nodes, live, rank }: { c: Compute; nodes: readonly Node[]; live: boolean; rank?: number }) {
  const navigate = useNavigate()
  const act = useStore((s) => s.act)
  const setUi = useStore((s) => s.setUi)
  const openSheet = useStore((s) => s.openSheet)
  const closeSheet = useStore((s) => s.closeSheet)
  const name = c.name ?? c.id

  const toActivity = (kind: 'logs' | 'events') => {
    setUi({ act: { ...act, kind, compute: c.id, rank: 'all' } })
    navigate('/activity')
  }
  const remove = async () => {
    await api.deleteCompute(c.id)
    closeSheet()
    navigate('/')
  }

  if (!live)
    return (
      <div className="spacer row" style={{ gap: 6 }}>
        <button className="btn sm" onClick={() => toActivity('events')}>
          <Icon name="events" />
          Events
        </button>
      </div>
    )
  return (
    <div className="spacer row" style={{ gap: 6, position: 'relative' }}>
      <button className="btn sm" onClick={() => openRun({ computeId: c.id, node: rank })}>
        <Icon name="run" />
        Run
      </button>
      <button className="btn sm" onClick={() => toActivity('logs')}>
        <Icon name="logs" />
        Logs
      </button>
      <button className="btn sm" onClick={() => openSheet({ kind: 'scale', computeId: c.id })}>
        <Icon name="scale" />
        Scale
      </button>
      <button
        className="btn sm"
        onClick={() => {
          setUi({ shell: true })
          navigate(`/computes/${c.id}/nodes/${rank ?? 0}`)
        }}
      >
        <Icon name="shell" />
        Shell
      </button>
      <button
        className="btn sm danger"
        onClick={() =>
          openSheet({
            kind: 'confirm',
            title: `Delete ${name}?`,
            body: `${nodes.filter(nodeLive).length} machines are terminated at the provider.`,
            confirm: 'Delete',
            danger: true,
            onConfirm: () => void remove(),
          })
        }
      >
        <Icon name="trash" />
        Delete
      </button>
    </div>
  )
}

/** The compute page's rail: a way back, and the actions. */
export function Rail() {
  const { id = '' } = useParams()
  const navigate = useNavigate()
  const c = useStore((s) => computeById(s, id))
  const live = useStore((s) => isLive(s, id))
  const nodes = useStore((s) => s.nodes[id]) ?? NONE
  if (!c) return null
  return (
    <>
      <div className="crumb">
        <button aria-label="Computes" onClick={() => navigate('/')}>
          <Icon name="fleet" />
          Computes
        </button>
      </div>
      <ComputeActions c={c} nodes={nodes} live={live} />
    </>
  )
}
