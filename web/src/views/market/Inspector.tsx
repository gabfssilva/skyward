import { useStore } from '../../state/store'
import { Icon } from '../../ui/icons'

export function Inspector() {
  const providers = useStore((s) => s.providers)
  const openSheet = useStore((s) => s.openSheet)
  return (
    <section className="card tight">
      <div className="cap">Where the offers come from</div>
      <div style={{ marginTop: 7 }}>
        {providers.map((p) => (
          <div className="kv" key={p.id}>
            <span>
              {p.kind}
              <span className="faint"> {p.name}</span>
            </span>
            <span className={p.last_error ? 'mono' : 'mono faint'} style={p.last_error ? { color: 'var(--bad)' } : undefined}>
              {p.last_error ? 'error' : p.offers_count.toLocaleString('en-US')}
            </span>
          </div>
        ))}
      </div>
      <button className="btn sm" style={{ marginTop: 10 }} onClick={() => openSheet({ kind: 'addProvider' })}>
        <Icon name="plus" />
        Add an account
      </button>
    </section>
  )
}
