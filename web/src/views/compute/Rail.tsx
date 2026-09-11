import type { ReactNode } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { api } from '../../api/client'
import { useStore, computeById, isLive } from '../../state/store'
import { valuesFor } from '../../state/nodes'
import { accrued, dur, gpusOf, median, money, ms, nodeLive, rateOf, readyOf, targetOf } from '../../state/model'
import { Icon } from '../../ui/icons'
import { Pill } from '../../ui/primitives'

const NONE: never[] = []

const Meter = ({ value, label, unit }: { value: ReactNode; label: string; unit?: string }) => (
  <div className="gauge-r">
    <b>
      {value}
      {unit ? <small>{unit}</small> : null}
    </b>
    <span>{label}</span>
  </div>
)

export function Rail({ computeId }: { computeId?: string }) {
  const params = useParams()
  const id = computeId ?? params.id ?? ''
  const navigate = useNavigate()
  const c = useStore((s) => computeById(s, id))
  const live = useStore((s) => isLive(s, id))
  const nodes = useStore((s) => s.nodes[id]) ?? NONE
  const metrics = useStore((s) => s.metrics)
  const openSheet = useStore((s) => s.openSheet)
  const closeSheet = useStore((s) => s.closeSheet)
  const setUi = useStore((s) => s.setUi)
  if (!c) return null

  const gpu = valuesFor(id, nodes, metrics, 'gpu')

  const remove = async () => {
    await api.deleteCompute(id)
    closeSheet()
    navigate('/computes')
  }

  return (
    <>
      <div className="gauge-r">
        <b style={{ fontSize: 28 }}>{c.name ?? c.id}</b>
        <span>
          <Pill state={c.status.state} /> · gen {c.generation} · {c.id}
        </span>
      </div>
      <Meter value={money(rateOf(nodes), 2)} label="per hour" unit="/h" />
      <Meter
        value={
          <>
            {readyOf(nodes).length}
            <small>/{targetOf(c)}</small>
          </>
        }
        label="nodes ready"
      />
      <Meter value={Math.round(median(gpu) || 0)} label="median GPU" unit="%" />
      <Meter value={gpusOf(nodes)} label="GPUs attached" />
      <Meter value={money(accrued(c, nodes), 0)} label={`spent in ${dur(Date.now() - ms(c.created_at))}`} />
      {live ? (
        <div className="spacer row" style={{ gap: 6, position: 'relative' }}>
          <button className="btn" onClick={() => openSheet({ kind: 'scale', computeId: id })}>
            <Icon name="scale" />
            Scale
          </button>
          <button className="btn" onClick={() => setUi({ dock: 'shell', dockMin: false })}>
            <Icon name="shell" />
            Shell
          </button>
          <button
            className="btn danger"
            onClick={() =>
              openSheet({
                kind: 'confirm',
                title: `Delete ${c.name ?? c.id}?`,
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
      ) : null}
    </>
  )
}
