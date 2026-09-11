import { useEffect } from 'react'
import { useParams } from 'react-router-dom'
import { api } from '../../api/client'
import { useStore, historyOf, computeById, isLive } from '../../state/store'
import { Legend } from '../../ui/primitives'
import { Comb, HexPop } from '../../ui/comb'
import { Icon } from '../../ui/icons'
import { combNodes } from '../../state/nodes'

const NONE: never[] = []

export function Stage() {
  const { id = '' } = useParams()
  const c = useStore((s) => computeById(s, id))
  const live = useStore((s) => isLive(s, id))
  const nodes = useStore((s) => s.nodes[id]) ?? NONE
  const metrics = useStore((s) => s.metrics)
  const progress = useStore((s) => s.progress)
  const sel = useStore((s) => s.sel)
  const choose = useStore((s) => s.pick)
  const setUi = useStore((s) => s.setUi)
  const loaded = nodes.length > 0
  useEffect(() => {
    if (!live && c && !loaded) void useStore.getState().reloadHistory(id)
  }, [live, c, loaded, id])
  if (!c) return null

  const selected = sel && sel.computeId === id ? sel.rank : null
  const pick = (rank: number) => choose(selected === rank ? null : { computeId: id, rank })
  const nodeAt = (rank: number) => nodes.find((n) => n.rank === rank)

  const shell = (rank: number) => setUi({ termNode: rank, dock: 'shell', dockMin: false })
  const drain = async (rank: number) => {
    const n = nodeAt(rank)
    if (!n) return
    choose(null)
    await api.drainNode(id, n.id)
    await useStore.getState().reloadCompute(id)
  }

  return (
    <section className="card">
      <Comb
        nodes={combNodes(id, nodes, metrics, progress)}
        computeId={id}
        name={c.name ?? c.id}
        selected={selected}
        onPick={pick}
        pop={(node, x, y, w) =>
          live ? (
            <HexPop
              node={node}
              x={x}
              y={y}
              w={w}
              history={historyOf(useStore.getState(), id, node.rank, 'gpu')}
              onShell={() => shell(node.rank)}
              onDrain={() => void drain(node.rank)}
              onReplace={() => void drain(node.rank)}
              onClose={() => choose(null)}
            />
          ) : (
            <HexPop node={node} x={x} y={y} w={w} onClose={() => choose(null)}>
              {node.error ? <div style={{ fontSize: '12.5px', lineHeight: 1.35 }}>{node.error}</div> : null}
              <div className="row" style={{ gap: 8, marginTop: 2 }}>
                <button className="hbtn" title="Close" onClick={() => choose(null)}>
                  <Icon name="close" />
                </button>
              </div>
            </HexPop>
          )
        }
      />
      <div style={{ marginTop: 14 }}>
        <Legend states={nodes.map((n) => n.state)} />
      </div>
    </section>
  )
}
