import { useState, type FormEvent } from 'react'
import { api } from '../api/client'
import { computeById, useStore } from '../state/store'
import { targetOf } from '../state/model'
import { Scrim, CloseBtn } from './Scrim'
import { COLLECTIVE } from './catalog'

export function Scale({ computeId }: { computeId: string }) {
  const compute = useStore((s) => computeById(s, computeId))
  const closeSheet = useStore((s) => s.closeSheet)
  const reload = useStore((s) => s.reloadCompute)
  const [busy, setBusy] = useState(false)
  if (!compute) return null

  const collective = compute.spec.plugins.filter((p) => COLLECTIVE.has(p.kind))
  const frozen = collective.length > 0

  const submit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    const form = new FormData(e.currentTarget)
    setBusy(true)
    try {
      await api.scale(compute.id, {
        initial: Number(form.get('initial')),
        min: Number(form.get('min')),
        max: compute.spec.nodes.max ?? null,
      })
      closeSheet()
      await reload(compute.id)
    } finally {
      setBusy(false)
    }
  }

  return (
    <Scrim>
      <div className="sheet" style={{ width: 'min(480px,100%)' }} role="dialog" aria-label="Scale">
        <div className="sheet-head">
          <b>Scale {compute.name}</b>
          <CloseBtn style={{ marginLeft: 'auto' }} />
        </div>
        <form className="sheet-body" style={{ display: 'grid', gap: 10 }} onSubmit={(e) => void submit(e)}>
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
              <input id="sc-initial" name="initial" type="number" min={0} defaultValue={targetOf(compute)} disabled={frozen} />
            </div>
            <div className="field">
              <label htmlFor="sc-min">Floor</label>
              <input id="sc-min" name="min" type="number" min={0} defaultValue={compute.spec.nodes.min ?? compute.spec.nodes.initial} disabled={frozen} />
            </div>
          </div>
          <div className="sub">A resize opens generation {compute.generation + 1}; nodes already ready are kept.</div>
          <button className="btn primary" type="submit" style={{ justifySelf: 'start' }} disabled={frozen || busy}>
            Apply
          </button>
        </form>
      </div>
    </Scrim>
  )
}
