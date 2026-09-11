import type { ReactNode } from 'react'
import { useStore } from '../../state/store'
import { median, money, rateOf } from '../../state/model'
import { Icon } from '../../ui/icons'
import { valuesFor } from '../../state/nodes'

function Meter({ value, label, unit }: { value: ReactNode; label: string; unit?: string }) {
  return (
    <div className="gauge-r">
      <b>
        {value}
        {unit ? <small>{unit}</small> : null}
      </b>
      <span>{label}</span>
    </div>
  )
}

/** The fleet rail: what the whole account is burning, and the way to buy more. */
export function Rail() {
  const computes = useStore((s) => s.computes)
  const nodesByCompute = useStore((s) => s.nodes)
  const metrics = useStore((s) => s.metrics)
  const openSheet = useStore((s) => s.openSheet)

  const ready = computes.flatMap((c) => (nodesByCompute[c.id] ?? []).filter((n) => n.state === 'ready').map((n) => ({ c, n })))
  const util = computes.flatMap((c) => valuesFor(c.id, nodesByCompute[c.id] ?? [], metrics, 'gpu'))
  const rate = computes.reduce((s, c) => s + rateOf(nodesByCompute[c.id] ?? []), 0)
  const gpus = computes.reduce((s, c) => s + (nodesByCompute[c.id] ?? []).filter((n) => n.state === 'ready').length * (c.spec.specs[0]?.accelerator_count ?? 1), 0)
  const idle = ready.filter(({ c, n }) => (metrics[`${c.id}/${n.rank}`]?.gpu ?? 0) < 25)
  const idleCost = idle.reduce((s, { n }) => s + (n.price_per_hour ?? 0), 0)

  return (
    <>
      <Meter value={money(rate, 2)} label="burning per hour" unit="/h" />
      <Meter value={ready.length} label={`nodes across ${computes.length} computes`} />
      <Meter value={gpus} label="GPUs attached" />
      <Meter value={Math.round(median(util) || 0)} label="median GPU" unit="%" />
      <Meter value={idle.length} label={`idling — ${money(idleCost)}/h wasted`} />
      <div className="spacer">
        <button className="btn primary" onClick={() => openSheet({ kind: 'wizard' })}>
          <Icon name="plus" />
          New compute
        </button>
      </div>
    </>
  )
}
