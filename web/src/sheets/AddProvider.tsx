import { useState, type FormEvent } from 'react'
import { api } from '../api/client'
import { useStore } from '../state/store'
import { Scrim, CloseBtn } from './Scrim'

export function AddProvider({ provider }: { provider?: string }) {
  const kinds = useStore((s) => s.providerKinds)
  const closeSheet = useStore((s) => s.closeSheet)
  const reload = useStore((s) => s.reloadProviders)
  const [kind, setKind] = useState(provider ?? kinds[0]?.kind ?? 'aws')
  const [busy, setBusy] = useState(false)
  const fields = kinds.find((k) => k.kind === kind)?.credential_fields ?? []

  const submit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    const form = new FormData(e.currentTarget)
    const credentials: Record<string, string> = {}
    for (const f of fields) credentials[f] = String(form.get(f) ?? '')
    setBusy(true)
    try {
      await api.createProvider({ kind, name: String(form.get('name')), credentials, config: {} })
      closeSheet()
      await reload()
    } finally {
      setBusy(false)
    }
  }

  return (
    <Scrim label="Add account">
      <div className="sheet" style={{ width: 'min(500px,100%)' }}>
        <div className="sheet-head">
          <b>Add account</b>
          <CloseBtn style={{ marginLeft: 'auto' }} />
        </div>
        <form id="add-provider" className="sheet-body" style={{ display: 'grid', gap: 10 }} onSubmit={(e) => void submit(e)}>
          <div className="field">
            <label htmlFor="pv-kind">Kind</label>
            <select id="pv-kind" name="kind" value={kind} onChange={(e) => setKind(e.target.value)}>
              {kinds.map((k) => (
                <option key={k.kind} value={k.kind}>
                  {k.kind}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor="pv-name">Account name</label>
            <input id="pv-name" name="name" defaultValue="default" required />
          </div>
          {fields.length ? (
            fields.map((f) => (
              <div className="field" key={f}>
                <label htmlFor={`pv-${f}`}>{f.replace(/_/g, ' ')}</label>
                <input id={`pv-${f}`} name={f} type="password" placeholder="••••••••" required />
              </div>
            ))
          ) : (
            <div className="sub">This kind needs no credentials.</div>
          )}
        </form>
        <div className="sheet-foot">
          <button className="btn primary" form="add-provider" type="submit" disabled={busy}>
            Add account
          </button>
        </div>
      </div>
    </Scrim>
  )
}
