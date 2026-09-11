import { useState } from 'react'
import { api, type Provider } from '../../api/client'
import { Icon } from '../../ui/icons'
import { Pill } from '../../ui/primitives'
import { useStore } from '../../state/store'

export function Stage() {
  const providers = useStore((s) => s.providers)
  const openSheet = useStore((s) => s.openSheet)
  const reloadProviders = useStore((s) => s.reloadProviders)
  const reloadOffers = useStore((s) => s.reloadOffers)
  const [busy, setBusy] = useState<string | null>(null)

  const fetchOffers = async (p: Provider) => {
    setBusy(p.id)
    try {
      await api.offers({ provider: p.id, refresh: true })
      await Promise.all([reloadOffers(), reloadProviders()])
    } finally {
      setBusy(null)
    }
  }

  const check = async (p: Provider) => {
    setBusy(p.id)
    try {
      await api.provider(p.id)
      await reloadProviders()
    } finally {
      setBusy(null)
    }
  }

  const remove = (p: Provider) =>
    openSheet({
      kind: 'confirm',
      title: 'Remove this account?',
      body: `${p.kind} ${p.name} will stop being priced, and computes it holds will keep running.`,
      confirm: 'Remove',
      danger: true,
      onConfirm: () => {
        void api.deleteProvider(p.id).then(reloadProviders)
      },
    })

  return (
    <section className="card">
      <div className="combhead">
        <b>Provider accounts</b>
        <button className="btn primary" style={{ marginLeft: 'auto' }} onClick={() => openSheet({ kind: 'addProvider' })}>
          <Icon name="plus" />
          Add account
        </button>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(280px,1fr))', gap: 10 }}>
        {providers.map((p) => (
          <div
            key={p.id}
            className={`strip ${p.last_error ? 'bad' : ''}`}
            style={{ flexDirection: 'column', alignItems: 'stretch', gap: 10, background: 'var(--sunk)' }}
          >
            <div className="row" style={{ justifyContent: 'space-between' }}>
              <div className="row" style={{ gap: 10 }}>
                <span style={{ fontSize: 22, color: 'var(--faint)', display: 'inline-flex' }}>
                  <Icon name="providers" />
                </span>
                <b style={{ fontWeight: 700, fontSize: 14 }}>{p.kind}</b>
                <span className="mono faint">{p.name}</span>
              </div>
              <Pill state={p.last_error ? 'failed' : 'ready'} />
            </div>
            {p.last_error ? (
              <div className="sub" style={{ color: 'var(--bad)' }}>
                {p.last_error.message}
              </div>
            ) : null}
            <div className="row" style={{ gap: 6 }}>
              <button className="btn sm" disabled={busy === p.id} onClick={() => void check(p)}>
                <Icon name="key" />
                Check
              </button>
              <button className="btn sm" disabled={busy === p.id} onClick={() => void fetchOffers(p)}>
                <Icon name="refresh" />
                Fetch offers
              </button>
              <button className="btn sm danger" style={{ marginLeft: 'auto' }} onClick={() => remove(p)}>
                <Icon name="trash" />
                Remove
              </button>
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}
