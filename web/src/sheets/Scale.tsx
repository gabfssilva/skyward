import { useState } from 'react'
import { api, type Compute } from '../api/client'
import { computeById, useStore } from '../state/store'
import { targetOf } from '../state/model'
import { Scrim, CloseBtn } from './Scrim'
import { COLLECTIVE } from './catalog'

export function Scale({ computeId }: { computeId: string }) {
  const compute = useStore((s) => computeById(s, computeId))
  return compute ? <Form compute={compute} /> : null
}

function Form({ compute }: { compute: Compute }) {
  const closeSheet = useStore((s) => s.closeSheet)
  const reload = useStore((s) => s.reloadCompute)
  const [busy, setBusy] = useState(false)
  const [nodes, setNodes] = useState(() => targetOf(compute))
  const [floor, setFloor] = useState(() => compute.spec.nodes.min ?? compute.spec.nodes.initial)

  const collective = compute.spec.plugins.filter((p) => COLLECTIVE.has(p.kind))
  const frozen = collective.length > 0

  const submit = async () => {
    setBusy(true)
    try {
      await api.scale(compute.id, { initial: nodes, min: floor, max: compute.spec.nodes.max ?? null })
      closeSheet()
      await reload(compute.id)
    } finally {
      setBusy(false)
    }
  }

  return (
    <Scrim label="Scale">
      <div className="sheet" style={{ width: 'min(480px,100%)' }}>
        <div className="sheet-head">
          <b>Scale {compute.name}</b>
          <CloseBtn style={{ marginLeft: 'auto' }} />
        </div>
        <div className="sheet-body" style={{ display: 'grid', gap: 10 }}>
          {frozen ? (
            <div className="strip" style={{ background: 'var(--warn-soft)', alignItems: 'flex-start', flexDirection: 'column', gap: 3 }}>
              <b style={{ fontWeight: 600 }}>This compute holds a collective plugin.</b>
              <span className="sub">
                {collective.map((p) => p.kind).join(', ')} freezes the world, so it cannot be resized. Open a new generation instead.
              </span>
            </div>
          ) : null}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
            <div className="field">
              <label htmlFor="sc-initial">Nodes</label>
              <input id="sc-initial" type="number" min={0} value={nodes} disabled={frozen} onChange={(e) => setNodes(Number(e.target.value))} />
            </div>
            <div className="field">
              <label htmlFor="sc-min">Floor</label>
              <input id="sc-min" type="number" min={0} value={floor} disabled={frozen} onChange={(e) => setFloor(Number(e.target.value))} />
            </div>
          </div>
          <div className="sub">A resize opens generation {compute.generation + 1}; nodes already ready are kept.</div>
        </div>
        <div className="sheet-foot">
          <button className="btn primary" disabled={frozen || busy} onClick={() => void submit()}>
            Scale to {nodes} node{nodes === 1 ? '' : 's'}
          </button>
        </div>
      </div>
    </Scrim>
  )
}
