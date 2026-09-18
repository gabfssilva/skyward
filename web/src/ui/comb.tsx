import { memo } from 'react'
import { SQ3, hexPts, hive, hiveSize } from '../state/model'

/**
 * One hexagon of a hive: which node it is, what colour it carries and what it says when pointed at.
 *
 * The drawing knows nothing about states or gauges — a page decides what a cell means and hands the
 * colour over, which is what lets the same hive read as nodes on one page and as executions on another.
 */
export type Hex = { rank: number; fill: string; tip: string; cls?: string; marked?: boolean }

/** The cell radius that fits the largest of these hives in the box, and so the scale every one of them is drawn at. */
export const scaleFor = (counts: readonly number[], room: number, tall: number): number =>
  Math.min(...counts.map((n) => hiveSize(n, room, tall)), 120)

/**
 * The cell radius that makes a hive of ``count`` fill a box of ``room`` × ``tall``, and the width it draws at.
 *
 * A hive is laid out from one radius and every coordinate in it scales with that radius, so a single probe
 * gives the scale exactly — the partial last ring included, which the ring count alone overstates. A page
 * gives the drawing's column that width: a compute of one node is then one large hexagon that owns its
 * half of the card, and one of three hundred is the same card filled with cells.
 */
export const fillRoom = (count: number, room: number, tall: number): { size: number; width: number } => {
  const probe = hive(Math.max(1, count), 100)
  const size = Math.min((room / probe.w) * 100, (tall / probe.h) * 100)
  return { size, width: Math.ceil((probe.w * size) / 100) }
}

type HiveProps = {
  cells: readonly Hex[]
  computeId: string
  /** the cell radius: the same number across hives that are meant to be compared */
  size: number
  label: string
  onPick?: (rank: number) => void
}

/**
 * A compute as one honeycomb: rings around a centre, a partial last ring standing on its base.
 *
 * Cells are positioned with a transform rather than by their points, so a node that appears or goes
 * away slides into place instead of being redrawn somewhere else.
 */
function Cells({ cells, computeId, size, label, onPick }: HiveProps) {
  const lay = hive(cells.length, size)
  const P = hexPts(size)
  const placed = cells.map((cell, i) => ({ cell, at: lay.cells[i]! }))
  /* an outlined cell is drawn last, so its stroke is not painted over by the cell next to it */
  const order = placed.some((p) => p.cell.marked) ? [...placed.filter((p) => !p.cell.marked), ...placed.filter((p) => p.cell.marked)] : placed
  return (
    <div className="comb" style={{ maxWidth: Math.ceil(lay.w) }}>
      <svg viewBox={`-1 -1 ${(lay.w + 2).toFixed(1)} ${(lay.h + 2).toFixed(1)}`} style={{ width: '100%' }} role="img" aria-label={label}>
        {order.map(({ cell, at }) => {
          return (
            <g
              key={cell.rank}
              className={['hx', cell.cls, cell.marked ? 'marked' : null].filter(Boolean).join(' ')}
              style={{ '--x': `${at[0].toFixed(1)}px`, '--y': `${at[1].toFixed(1)}px` }}
              data-id={computeId}
              data-rank={cell.rank}
              data-tip={cell.tip}
              onClick={() => onPick?.(cell.rank)}
            >
              <polygon className="cell" points={P} fill={cell.fill} />
              <polygon className="edge" points={P} />
            </g>
          )
        })}
      </svg>
    </div>
  )
}

/** A gauge moves every couple of seconds and a compute has hundreds of cells, so a redraw is worth it only where a colour changed. */
const shape = (p: HiveProps): string => `${p.computeId}|${p.size}|${p.cells.map((c) => `${c.rank}${c.fill}${c.cls ?? ''}${c.marked ? '*' : ''}`).join(',')}`

export const Hive = memo(Cells, (before, after) => shape(before) === shape(after))

/** One hexagon per worker slot, lit while an execution occupies it. */
export function Slots({ slots, busy, size = 13, tips = [] }: { slots: number; busy: number; size?: number; tips?: readonly string[] }) {
  const P = hexPts(size)
  const step = size * SQ3 * 1.1
  const w = step * slots
  const h = size * 2
  return (
    <svg className="slothive" viewBox={`-1 -1 ${(w + 2).toFixed(1)} ${(h + 2).toFixed(1)}`} style={{ width: Math.ceil(w), flex: 'none' }} role="img" aria-label={`${busy} of ${slots} slots busy`}>
      {Array.from({ length: slots }, (_, i) => (
        <g key={i} transform={`translate(${(i * step + step / 2).toFixed(1)},${(h / 2).toFixed(1)})`} data-tip={i < busy ? (tips[i] ?? 'busy') : 'idle'}>
          <polygon className={i < busy ? 'slot on' : 'slot'} points={P} />
          {size >= 13 ? <text y="4">{i}</text> : null}
        </g>
      ))}
    </svg>
  )
}
