import { useNavigate } from 'react-router-dom'
import { Icon } from '../../ui/icons'
import { Pill } from '../../ui/primitives'
import { ComputeActions, ComputeStats } from '../compute/Rail'
import { useNode } from './Stage'

/** The node page's compact rail: the full crumb, the compute's stats, and its actions. */
export function Rail() {
  const navigate = useNavigate()
  const found = useNode()
  if (!found) return null
  const { c, n, live, nodes } = found
  return (
    <>
      <div className="crumb">
        <button aria-label="Computes" onClick={() => navigate('/')}>
          <Icon name="fleet" />
        </button>
        <span className="sep">/</span>
        <button onClick={() => navigate(`/computes/${c.id}`)}>
          <b>{c.name ?? c.id}</b>
        </button>
        <Pill state={c.status.state} />
        <span className="sep">/</span>
        <span className="mono">rank {n.rank}</span>
        <Pill state={n.state} />
      </div>
      <ComputeStats c={c} nodes={nodes} live={live} />
      <ComputeActions c={c} nodes={nodes} live={live} rank={n.rank} />
    </>
  )
}
