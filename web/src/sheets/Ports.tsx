import type { FormEvent } from 'react'
import { computeById, nodesOf, useStore } from '../state/store'
import { Scrim, CloseBtn } from './Scrim'
import { usePorts } from './port-state'

export function Ports({ computeId }: { computeId: string }) {
  const compute = useStore((s) => computeById(s, computeId))
  const nodes = useStore((s) => nodesOf(s, computeId))
  const ports = usePorts((s) => s.ports[computeId] ?? [])
  const add = usePorts((s) => s.add)
  const close = usePorts((s) => s.close)
  if (!compute) return null

  const submit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    const form = new FormData(e.currentTarget)
    add(computeId, {
      route: String(form.get('route')),
      remote: Number(form.get('remote')),
      local: Number(form.get('local')),
      node: Number(form.get('node')),
      state: 'open',
    })
    e.currentTarget.reset()
  }

  return (
    <Scrim label="Ports">
      <div className="sheet" style={{ width: 'min(560px,100%)' }}>
        <div className="sheet-head">
          <b>Ports on {compute.name}</b>
          <CloseBtn style={{ marginLeft: 'auto' }} />
        </div>
        <div className="sheet-body" style={{ display: 'grid', gap: 12 }}>
          {ports.length ? (
            ports.map((p) => (
              <div className="strip" style={{ background: 'var(--sunk)' }} key={p.remote}>
                <b style={{ fontWeight: 600 }}>{p.route}</b>
                <span className="mono faint">
                  rank {p.node} · :{p.remote}
                </span>
                <span className="mono" style={{ color: 'var(--accent)' }}>
                  127.0.0.1:{p.local}
                </span>
                <button className="btn sm danger" style={{ marginLeft: 'auto' }} onClick={() => close(computeId, p.remote)}>
                  Close
                </button>
              </div>
            ))
          ) : (
            <div className="sub">No port is bridged yet.</div>
          )}
          <form className="portform" onSubmit={submit}>
            <div className="field">
              <label htmlFor="pt-route">Route</label>
              <input id="pt-route" name="route" placeholder="inference" required />
            </div>
            <div className="field">
              <label htmlFor="pt-remote">Node port</label>
              <input id="pt-remote" name="remote" type="number" defaultValue={8080} required />
            </div>
            <div className="field">
              <label htmlFor="pt-local">Local port</label>
              <input id="pt-local" name="local" type="number" defaultValue={8080} required />
            </div>
            <div className="field">
              <label htmlFor="pt-node">Rank</label>
              <input id="pt-node" name="node" type="number" defaultValue={0} min={0} max={Math.max(0, nodes.length - 1)} />
            </div>
            <button className="btn primary" type="submit">
              Forward
            </button>
          </form>
        </div>
      </div>
    </Scrim>
  )
}
