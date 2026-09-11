import { useNavigate } from 'react-router-dom'
import { api } from '../../api/client'
import type { Compute, Node } from '../../api/client'
import { useStore, historyOf } from '../../state/store'
import { Comb, HexPop } from '../../ui/comb'
import { combNodes } from '../../state/nodes'

/**
 * A compute's honeycomb, wired to the store's selection.
 *
 * Picking a hex toggles the selection and moves to that compute, the way the
 * prototype's ``pick`` action does; the raised hex carries the in-place detail.
 */
export function LiveComb({ compute, nodes, room, cols, size }: { compute: Compute; nodes: readonly Node[]; room?: number; cols?: number; size?: number }) {
  const navigate = useNavigate()
  const metrics = useStore((s) => s.metrics)
  const sel = useStore((s) => s.sel)
  const pick = useStore((s) => s.pick)
  const setUi = useStore((s) => s.setUi)
  const cells = combNodes(compute.id, nodes, metrics)
  const selected = sel && sel.computeId === compute.id ? sel.rank : null

  const onPick = (rank: number) => {
    pick(sel && sel.computeId === compute.id && sel.rank === rank ? null : { computeId: compute.id, rank })
    navigate(`/computes/${compute.id}`)
  }

  const nodeAt = (rank: number) => nodes.find((n) => n.rank === rank)

  return (
    <Comb
      nodes={cells}
      computeId={compute.id}
      name={compute.name ?? undefined}
      selected={selected}
      onPick={onPick}
      room={room}
      cols={cols}
      size={size}
      pop={(node, x, y, w) => (
        <HexPop
          node={node}
          x={x}
          y={y}
          w={w}
          history={historyOf(useStore.getState(), compute.id, node.rank, 'gpu')}
          onClose={() => pick(null)}
          onShell={() => {
            setUi({ termNode: node.rank, dock: 'shell', dockMin: false })
            navigate(`/computes/${compute.id}`)
          }}
          onDrain={() => {
            const n = nodeAt(node.rank)
            if (n) void api.drainNode(compute.id, n.id).then(() => useStore.getState().reloadCompute(compute.id))
          }}
          onReplace={() => {
            const n = nodeAt(node.rank)
            if (n) void api.drainNode(compute.id, n.id).then(() => useStore.getState().reloadCompute(compute.id))
          }}
        />
      )}
    />
  )
}
