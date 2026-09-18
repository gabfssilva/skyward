import type { CSSProperties } from 'react'
import type { Node } from '../../api/client'
import { holdersOf } from '../../state/model'
import type { PhaseMark } from '../../state/nodes'
import { useStore } from '../../state/store'

const CHIP: Record<PhaseMark['state'], CSSProperties> = {
  completed: { color: 'var(--ok)', background: 'var(--ok-soft)' },
  started: { color: 'var(--boot)', background: 'var(--boot-soft)' },
  failed: { color: 'var(--bad)', background: 'var(--bad-soft)' },
}

/**
 * How far a machine that is still coming up has got: a fraction while the provider reports one, then the
 * phases the machine itself has reached. Nothing is drawn until the node has said something.
 *
 * Which phases a bootstrap runs depends on the image and the plugins, so the checklist is whatever the node
 * has said so far rather than a list written here.
 */
export function PhaseProgress({ nodeId, width }: { nodeId: string; width?: number }) {
  const p = useStore((s) => s.progress[nodeId])
  if (!p || (!p.phase && p.completion == null && !p.phases.length)) return null
  return (
    <>
      {p.phase ? <div className="cap">{p.phase}</div> : null}
      {p.completion != null ? (
        <div className="track" style={{ marginTop: 6, width, maxWidth: '100%' }}>
          <i style={{ width: `${p.completion * 100}%`, background: 'var(--boot)' }} />
        </div>
      ) : null}
      {p.phases.length ? (
        <div className="chips" style={{ marginTop: 9 }}>
          {p.phases.map((ph) => (
            <span key={ph.name} className="chip" style={{ height: 22, fontSize: 11, ...CHIP[ph.state] }}>
              {ph.state === 'completed' ? '✓ ' : ''}
              {ph.name}
            </span>
          ))}
        </div>
      ) : null}
    </>
  )
}

/** The node of a compute that is furthest into its bootstrap, which is the one worth showing the progress of. */
export function Booting({ nodes }: { nodes: readonly Node[] }) {
  const progress = useStore((s) => s.progress)
  const coming = holdersOf(nodes).filter((n) => n.state !== 'deleted' && n.state !== 'failed' && n.state !== 'lost')
  const furthest = coming.slice().sort((a, b) => (progress[b.id]?.phases.length ?? 0) - (progress[a.id]?.phases.length ?? 0))[0]
  if (!furthest) return null
  return <PhaseProgress nodeId={furthest.id} width={380} />
}
