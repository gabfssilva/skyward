import { memo, type CSSProperties } from 'react'
import { KIND, KIND_FILL, SQ3, clamp, hexCols, hexPts, hexSize, hive, hiveSize, money, ringsFor, spiral, type NodeMetrics } from '../state/model'

export type CombNode = {
  rank: number
  state: string
  address?: string | null
  price?: number | null
  phase?: string | null
  error?: string | null
  m?: Partial<NodeMetrics>
}

/** A ready cell is the accent, deepened by how hard its GPU is working; any other cell is its state's colour. */
export const tone = (node: CombNode): string => {
  const k = KIND(node.state)
  if (k !== 'ready') return KIND_FILL[k]
  const gpu = clamp(node.m?.gpu ?? 0, 0, 100)
  return `color-mix(in oklab, var(--ok) ${Math.round(38 + gpu * 0.62)}%, var(--panel))`
}

const tipOf = (node: CombNode): string =>
  node.state === 'ready'
    ? `rank ${node.rank} · gpu ${Math.round(node.m?.gpu ?? 0)}% · ${node.address ?? '—'} · ${money(node.price ?? 0)}/h`
    : `rank ${node.rank} · ${node.state}${node.error ? ' — ' + node.error : node.phase ? ' — ' + node.phase : ''}`

type CombProps = {
  nodes: readonly CombNode[]
  computeId: string
  name?: string
  onPick?: (rank: number) => void
  cols?: number
  size?: number
  room?: number
  only?: ReadonlySet<number>
  /** ``hive`` lays the cells out as the prototype's ``comb`` does: rings around a centre, ``size`` being the cell's circumradius */
  layout?: 'grid' | 'hive'
}

type Geometry = { W: number; H: number; width: number; height: number; pos: (i: number) => [number, number] }

/** Rows of hexes, every other row shifted by half a cell. */
const gridGeometry = (n: number, forcedCols?: number, forcedSize?: number, room?: number): Geometry => {
  const cols = forcedCols || hexCols(n)
  const rows = Math.ceil(n / cols)
  const W = forcedSize || hexSize(n, cols, room)
  const H = W * 1.1547
  const gap = Math.max(1.5, W * 0.1)
  const dx = W + gap
  const dy = H * 0.75 + gap * 0.87
  return {
    W,
    H,
    width: cols * dx + (rows > 1 ? dx / 2 : 0),
    height: (rows - 1) * dy + H,
    pos: (i) => {
      const row = Math.floor(i / cols)
      return [(i % cols) * dx + (row % 2 ? dx / 2 : 0), row * dy]
    },
  }
}

/** Rings around a centre cell, a partial last ring standing on its base. */
const hiveGeometry = (n: number, s: number): Geometry => {
  const lay = hive(n, s)
  const W = s * SQ3
  const H = 2 * s
  return { W, H, width: lay.w, height: lay.h, pos: (i) => [lay.cells[i]![0] - W / 2, lay.cells[i]![1] - H / 2] }
}

function Hexes({
  nodes,
  computeId,
  name,
  onPick,
  cols: forcedCols,
  size: forcedSize,
  room,
  only,
  layout = 'grid',
}: CombProps) {
  const n = nodes.length
  const { W, H, width, height, pos } = layout === 'hive' ? hiveGeometry(n, forcedSize || hiveSize(n, room ?? 980, room ?? 640)) : gridGeometry(n, forcedCols, forcedSize, room)
  const P = `${W / 2},0 ${W},${(H / 4).toFixed(2)} ${W},${((H * 3) / 4).toFixed(2)} ${W / 2},${H.toFixed(2)} 0,${((H * 3) / 4).toFixed(2)} 0,${(H / 4).toFixed(2)}`
  return (
    <div className="comb" style={{ maxWidth: Math.ceil(width), '--w': `${Math.ceil(width)}px` } as CSSProperties}>
      <svg viewBox={`-1 -1 ${(width + 2).toFixed(1)} ${(height + 2).toFixed(1)}`} style={{ width: '100%' }} role="img" aria-label={`${n} nodes${name ? ' of ' + name : ''}`}>
        {nodes.map((node, i) => {
          const [x, y] = pos(i)
          const k = KIND(node.state)
          return (
            <g
              key={node.rank}
              className={`hx ${k}`}
              style={{ '--x': `${x.toFixed(1)}px`, '--y': `${y.toFixed(1)}px` } as CSSProperties}
              data-id={computeId}
              data-rank={node.rank}
              data-tip={tipOf(node)}
              onClick={() => onPick?.(node.rank)}
            >
              <polygon className="cell" points={P} fill={only != null && !only.has(node.rank) ? 'var(--sunk)' : layout === 'hive' ? tone(node) : KIND_FILL[k]} />
              <polygon className="edge" points={P} />
            </g>
          )
        })}
      </svg>
    </div>
  )
}

/**
 * A gauge moves every couple of seconds and a compute has hundreds of hexes; what
 * a hex draws is its state, not its gauges, so a gauge is worth a redraw only
 * where it changes a hive cell's tone.
 */
const drawn = (node: CombNode, hive: boolean): string =>
  `${node.rank}|${node.state}|${node.error ?? ''}|${node.phase ?? ''}` + (hive ? `|${tone(node)}` : '')

const shape = (props: CombProps): string =>
  [props.computeId, props.cols ?? '', props.size ?? '', props.room ?? '', props.layout ?? '', props.only ? [...props.only].join('.') : '']
    .concat(props.nodes.map((n) => drawn(n, props.layout === 'hive')))
    .join(';')

export const Comb = memo(Hexes, (before, after) => shape(before) === shape(after))

/* ---------- the fleet: every compute as one hive of the same outer size ---------- */

export type HiveItem = {
  id: string
  name: string
  state: string
  /** what the compute costs per hour */
  rate: number
  /** worker slots per node */
  slots: number
  nodes: readonly CombNode[]
  /** running executions per rank */
  busy: Readonly<Record<number, number>>
}

type Axial = readonly [number, number]

const rowOf = (n: number): Axial[] => Array.from({ length: n }, (_, i) => [i, 0] as const)

const bounds = (cells: readonly Axial[]) => {
  const xs = cells.map(([q, r]) => SQ3 * (q + r / 2))
  const ys = cells.map(([, r]) => 1.5 * r)
  const minX = Math.min(...xs)
  const minY = Math.min(...ys)
  return { minX, minY, w: Math.max(...xs) - minX + SQ3, h: Math.max(...ys) - minY + 2 }
}

const GAP = 1.1
const LABEL = 0.22

const busyLabel = (item: HiveItem, rank: number): string => {
  const busy = item.busy[rank] ?? 0
  return item.slots > 1 ? `${busy}/${item.slots} slots` : busy ? 'busy' : 'idle'
}

/**
 * The prototype's ``hives``: every compute drawn as one honeycomb region of the same
 * outer size, laid out in a row with a caption under each, or — when a row would make
 * them too small — in a spiral with the captions in a row below the drawing.
 */
export function Hives({
  items,
  room = 1010,
  tall = 470,
  onOpen,
}: {
  items: readonly HiveItem[]
  room?: number
  tall?: number
  onOpen?: (computeId: string) => void
}) {
  const order = items.slice().sort((a, b) => b.nodes.length - a.nodes.length)
  const C = order.length
  if (!C) return null
  const fits = [
    { cells: rowOf(C), cap: 1 },
    { cells: spiral(C), cap: 0 },
  ].map(({ cells, cap }) => {
    const bb = bounds(cells)
    return { cells, bb, cap, R: Math.min(room / (bb.w * GAP), tall / ((bb.h + cap * LABEL) * GAP)) }
  })
  const pick = fits.sort((x, y) => y.R - x.R)[0]!
  const R = Math.min(170, pick.R)
  const stepB = R * GAP
  const W = pick.bb.w * stepB
  const H = (pick.bb.h + pick.cap * LABEL) * stepB
  const at = ([q, r]: Axial): [number, number] => [(SQ3 * (q + r / 2) - pick.bb.minX + SQ3 / 2) * stepB, (1.5 * r - pick.bb.minY + 1) * stepB]

  const groups = order.map((item, i) => {
    const [cx, cy] = at(pick.cells[i]!)
    const n = item.nodes.length
    const k = ringsFor(n)
    const s = (((R / (2 * k + 1.12) / 1.12) * 2) / SQ3) * 0.96
    const lay = hive(n, s)
    const P = hexPts(s)
    return (
      <g className="region" key={item.id}>
        {item.nodes.map((node, j) => {
          const [x, y] = lay.cells[j]!
          const kind = KIND(node.state)
          const tip = `${item.name} · rank ${node.rank} · ${node.state === 'ready' ? `gpu ${Math.round(node.m?.gpu ?? 0)}% · ${busyLabel(item, node.rank)}` : node.state}`
          const style = { '--x': `${(cx + x - lay.w / 2).toFixed(1)}px`, '--y': `${(cy + y - lay.h / 2).toFixed(1)}px` } as CSSProperties
          return (
            <g key={node.rank} className={`hx ${kind}`} style={style} data-id={item.id} data-rank={node.rank} data-tip={tip} onClick={() => onOpen?.(item.id)}>
              <polygon className="cell" points={P} fill={tone(node)} />
              <polygon className="edge" points={P} />
            </g>
          )
        })}
        {pick.cap ? (
          <g className="hcap" transform={`translate(${cx.toFixed(1)},${(cy + R * 0.92 + 14).toFixed(1)})`} onClick={() => onOpen?.(item.id)}>
            <text className="hname" y="4">
              {item.name}
            </text>
            <text className="hmeta" y="19">
              {n} node{n === 1 ? '' : 's'} · {money(item.rate)}/h
            </text>
          </g>
        ) : null}
      </g>
    )
  })

  return (
    <>
      <div className="comb" style={{ maxWidth: Math.ceil(W), margin: '0 auto' }}>
        <svg viewBox={`-2 -2 ${(W + 4).toFixed(1)} ${(H + 4).toFixed(1)}`} style={{ width: '100%' }} role="img" aria-label={`${C} computes`}>
          {groups}
        </svg>
      </div>
      {pick.cap ? null : (
        <div className="hivecaps">
          {order.map((item) => {
            const ready = item.nodes.filter((x) => x.state === 'ready')
            const busy = ready.reduce((a, x) => a + (item.busy[x.rank] ?? 0), 0)
            return (
              <button key={item.id} className="hivecap" onClick={() => onOpen?.(item.id)}>
                <i className={`dot ${item.state}`} />
                <b>{item.name}</b>
                <span className="mono faint">
                  {item.nodes.length} nodes · {item.slots > 1 ? `${busy}/${ready.length * item.slots} slots` : `${busy} busy`} · {money(item.rate)}/h
                </span>
              </button>
            )
          })}
        </div>
      )}
    </>
  )
}
