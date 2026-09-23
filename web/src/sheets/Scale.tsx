import { useState } from 'react'
import { api, type Compute } from '../api/client'
import { computeById, useStore } from '../state/store'
import { rangeOf } from '../state/model'
import { Num } from '../ui/primitives'
import { Scrim, CloseBtn } from './Scrim'
import { COLLECTIVE } from './catalog'

export function Scale({ computeId }: { computeId: string }) {
  const compute = useStore((s) => computeById(s, computeId))
  return compute ? <Form compute={compute} /> : null
}

/**
 * The size a compute is held to: the floor it stands on, and the ceiling it may grow to.
 *
 * ``initial`` is not here and is sent back untouched. It is the size the pool opened at, asked for once when it was
 * created, and nothing after the creation gets to reopen a pool that is already standing. What a resize moves is the
 * range — which is also what the reconciler holds the pool within, so the number typed here is the number it keeps.
 *
 * Both bounds are written, never one: leaving the ceiling out is what lets ``initial`` come back as the other end of
 * the range, and a pool asked for fifty under a ceiling of twenty is the one thing this form must not be able to say.
 */
function Form({ compute }: { compute: Compute }) {
  const closeSheet = useStore((s) => s.closeSheet)
  const reload = useStore((s) => s.reloadCompute)
  const held = rangeOf(compute)
  const [busy, setBusy] = useState(false)
  const [nodes, setNodes] = useState(held.floor)
  const [upTo, setUpTo] = useState(held.elastic ? held.ceiling : null)

  const collective = compute.spec.plugins.filter((p) => COLLECTIVE.has(p.kind))
  const frozen = collective.length > 0
  const inverted = upTo !== null && upTo < nodes
  const asked = upTo === null || upTo === nodes ? `${nodes} node${nodes === 1 ? '' : 's'}` : `${nodes}–${upTo} nodes`

  const submit = async () => {
    setBusy(true)
    try {
      await api.scale(compute.id, { initial: compute.spec.nodes.initial, min: nodes, max: upTo ?? nodes })
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
            <Num id="sc-nodes" label="Nodes" min={0} value={nodes} onChange={setNodes} />
            <Num id="sc-up" label="Up to" min={0} placeholder="not elastic" optional value={upTo} onChange={setUpTo} />
          </div>
          {inverted ? (
            <div className="strip" style={{ background: 'var(--warn-soft)' }}>
              <span className="sub">A ceiling of {upTo} is below the {nodes} asked for, and the ceiling is what the pool would be held to.</span>
            </div>
          ) : (
            <div className="sub">
              A resize opens generation {compute.generation + 1}; nodes already ready are kept. It opened at {compute.spec.nodes.initial}, which only its
              creation decides.
            </div>
          )}
        </div>
        <div className="sheet-foot">
          <button className="btn primary" disabled={frozen || busy || inverted} onClick={() => void submit()}>
            Scale to {asked}
          </button>
        </div>
      </div>
    </Scrim>
  )
}
