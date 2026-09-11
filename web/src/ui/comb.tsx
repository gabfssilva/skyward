import { memo, useEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react'
import { KIND, KIND_FILL, PHASES, UNIT, clamp, dur, hexCols, hexSize, money } from '../state/model'
import { Icon } from './icons'
import { Spark } from './charts'

export type CombNode = {
  rank: number
  state: string
  id?: string
  address?: string | null
  price?: number | null
  market?: string | null
  launched_at?: number | null
  phase?: string | null
  completion?: number | null
  phases_done?: number | null
  error?: string | null
  m?: Partial<Record<'gpu' | 'vram' | 'cpu' | 'temp' | 'net', number>>
}

const tipOf = (node: CombNode): string =>
  node.state === 'ready'
    ? `rank ${node.rank} · gpu ${Math.round(node.m?.gpu ?? 0)}% · ${node.address ?? '—'} · ${money(node.price ?? 0)}/h`
    : `rank ${node.rank} · ${node.state}${node.error ? ' — ' + node.error : node.phase ? ' — ' + node.phase : ''}`

type CombProps = {
  nodes: readonly CombNode[]
  computeId: string
  name?: string
  selected?: number | null
  onPick?: (rank: number) => void
  cols?: number
  size?: number
  room?: number
  only?: ReadonlySet<number>
  pop?: (node: CombNode, x: number, y: number, w: number) => ReactNode
}

function Hexes({
  nodes,
  computeId,
  name,
  selected,
  onPick,
  cols: forcedCols,
  size: forcedSize,
  room,
  only,
  pop,
}: CombProps) {
  const n = nodes.length
  const cols = forcedCols || hexCols(n)
  const rows = Math.ceil(n / cols)
  const W = forcedSize || hexSize(n, cols, room)
  const H = W * 1.1547
  const gap = Math.max(1.5, W * 0.1)
  const dx = W + gap
  const dy = H * 0.75 + gap * 0.87
  const width = cols * dx + (rows > 1 ? dx / 2 : 0)
  const height = (rows - 1) * dy + H
  const P = `${W / 2},0 ${W},${(H / 4).toFixed(2)} ${W},${((H * 3) / 4).toFixed(2)} ${W / 2},${H.toFixed(2)} 0,${((H * 3) / 4).toFixed(2)} 0,${(H / 4).toFixed(2)}`
  const pos = (i: number): [number, number] => {
    const row = Math.floor(i / cols)
    return [(i % cols) * dx + (row % 2 ? dx / 2 : 0), row * dy]
  }
  const si = selected == null ? -1 : nodes.findIndex((x) => x.rank === selected)
  const S = si >= 0 ? clamp(270 / W, 2.5, 9) : 1
  const R = (W * S) / 2
  const [sx, sy] = si >= 0 ? pos(si) : [0, 0]
  const scx = width > 2 * R ? clamp(sx + W / 2, R, width - R) : width / 2
  const scy = height > 2 * R * 1.1547 ? clamp(sy + H / 2, R * 1.1547, height - R * 1.1547) : height / 2

  const [open, setOpen] = useState(false)
  const previous = useRef<number | null | undefined>(null)
  useEffect(() => {
    if (si < 0) {
      previous.current = null
      setOpen(false)
      return
    }
    if (previous.current === selected) return
    previous.current = selected
    setOpen(false)
    const frame = requestAnimationFrame(() => setOpen(true))
    return () => cancelAnimationFrame(frame)
  }, [si, selected])

  const cells = nodes.map((node, i) => {
    const [x, y] = pos(i)
    const cx = x + W / 2
    const cy = y + H / 2
    const k = KIND(node.state)
    const out = only != null && !only.has(node.rank)
    const under = si >= 0 && i !== si && Math.hypot(cx - scx, cy - scy) < R + W * 0.45
    const x2 = i === si ? scx - W / 2 : x
    const y2 = i === si ? scy - H / 2 : y
    const style = {
      '--x': `${x.toFixed(1)}px`,
      '--y': `${y.toFixed(1)}px`,
      '--x2': `${x2.toFixed(1)}px`,
      '--y2': `${y2.toFixed(1)}px`,
      '--s': S,
    } as CSSProperties
    return (
      <g
        key={node.rank}
        className={`hx ${k}${i === si ? ' sel' : ''}${under ? ' under' : ''}`}
        style={style}
        data-id={computeId}
        data-rank={node.rank}
        data-tip={tipOf(node)}
        onClick={() => onPick?.(node.rank)}
      >
        <polygon className="cell" points={P} fill={out ? 'var(--sunk)' : KIND_FILL[k]} />
        <polygon className="edge" points={P} />
      </g>
    )
  })
  if (si >= 0) cells.push(cells.splice(si, 1)[0]!)

  const sel = si >= 0 ? nodes[si]! : null
  return (
    <div className="comb" style={{ maxWidth: Math.ceil(width) }}>
      <svg
        className={si >= 0 && open ? 'open' : ''}
        viewBox={`-1 -1 ${(width + 2).toFixed(1)} ${(height + 2).toFixed(1)}`}
        style={{ width: '100%' }}
        role="img"
        aria-label={`${n} nodes${name ? ' of ' + name : ''}`}
      >
        {cells}
      </svg>
      {sel && pop ? pop(sel, ((scx + 1) / (width + 2)) * 100, ((scy + 1) / (height + 2)) * 100, ((W * S) / (width + 2)) * 100) : null}
    </div>
  )
}

/**
 * A gauge moves every couple of seconds and a compute has hundreds of hexes; what
 * a hex draws is its state, not its gauges. Only the raised one reads the numbers,
 * so only the raised one is worth a redraw when they change.
 */
const drawn = (node: CombNode, selected: boolean): string =>
  `${node.rank}|${node.state}|${node.error ?? ''}|${node.phase ?? ''}|${node.completion ?? ''}|${node.phases_done ?? ''}` +
  (selected ? `|${JSON.stringify(node.m ?? {})}|${node.address ?? ''}|${node.price ?? ''}` : '')

const shape = (props: CombProps): string =>
  [props.computeId, props.selected ?? '', props.cols ?? '', props.size ?? '', props.room ?? '', props.only ? [...props.only].join('.') : '']
    .concat(props.nodes.map((n) => drawn(n, n.rank === props.selected)))
    .join(';')

export const Comb = memo(Hexes, (before, after) => shape(before) === shape(after))

export function HexPop({
  node,
  x,
  y,
  w,
  history,
  onShell,
  onDrain,
  onReplace,
  onClose,
  children,
}: {
  node: CombNode
  x: number
  y: number
  w: number
  history?: readonly number[]
  onShell?: () => void
  onDrain?: () => void
  onReplace?: () => void
  onClose?: () => void
  children?: ReactNode
}) {
  const body =
    children ??
    (node.state === 'ready' ? (
      <>
        <div className="trio">
          {(
            [
              ['gpu', 'GPU'],
              ['vram', 'VRAM'],
              ['temp', 'Temp'],
            ] as const
          ).map(([m, l]) => (
            <div key={m}>
              <b>
                {Math.round(node.m?.[m] ?? 0)}
                {UNIT[m]}
              </b>
              <span>{l}</span>
            </div>
          ))}
        </div>
        <Spark values={history ?? []} h={30} fmt={(v) => Math.round(v) + '%'} />
        <div className="mono" style={{ opacity: 0.85 }}>
          {node.address} · {money(node.price ?? 0)}/h {node.market === 'spot' ? 'spot' : 'on demand'} · up {dur(Date.now() - (node.launched_at ?? Date.now()))}
        </div>
        <div className="row" style={{ gap: 8, marginTop: 2 }}>
          <button className="hbtn" title="Open a shell" onClick={onShell}>
            <Icon name="shell" />
          </button>
          <button className="hbtn" title="Drain" onClick={onDrain}>
            <Icon name="hide" />
          </button>
          <button className="hbtn" title="Close" onClick={onClose}>
            <Icon name="close" />
          </button>
        </div>
      </>
    ) : node.error ? (
      <>
        <div style={{ fontSize: '12.5px', lineHeight: 1.35 }}>{node.error}</div>
        <div className="row" style={{ gap: 8, marginTop: 2 }}>
          <button className="hbtn" title="Replace it" onClick={onReplace}>
            <Icon name="refresh" />
          </button>
          <button className="hbtn" title="Close" onClick={onClose}>
            <Icon name="close" />
          </button>
        </div>
      </>
    ) : (
      <>
        <div className="mono" style={{ opacity: 0.85 }}>
          {node.phase || 'waiting'}
        </div>
        <div className="track" style={{ width: '70%', background: 'rgba(0,0,0,.18)' }}>
          <i style={{ width: `${(node.completion ?? 0) * 100}%`, background: 'currentColor' }} />
        </div>
        <div className="mono" style={{ opacity: 0.85 }}>
          {node.phases_done ?? 0} of {PHASES.length} phases
        </div>
        <div className="row" style={{ gap: 8, marginTop: 2 }}>
          <button className="hbtn" title="Close" onClick={onClose}>
            <Icon name="close" />
          </button>
        </div>
      </>
    ))
  return (
    <div className="hexpop" style={{ left: `${x.toFixed(2)}%`, top: `${y.toFixed(2)}%`, width: `${w.toFixed(2)}%` }} role="dialog" aria-label={`rank ${node.rank}`}>
      <div className="in">
        <div className="mono" style={{ opacity: 0.85 }}>
          rank {node.rank} · {node.id}
        </div>
        <b className="st">{node.state.replace(/_/g, ' ')}</b>
        {body}
      </div>
    </div>
  )
}
