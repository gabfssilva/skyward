import { clamp } from '../state/model'
import type { Marks } from '../state/metrics'

/**
 * One metric over time: a line, the spread around it, and a second line behind it.
 *
 * Every chart on a page is handed the same ``axis``, so a checkpoint that slowed the GPUs down is
 * the same column in utilisation, power, CPU and whatever the image measures of its own. The scale
 * is the gauge's, not the data's: a percentage is drawn 0 to 100 and a memory against its capacity,
 * so the same chart at two moments is the same picture.
 */
export function Plot({ marks, axis, w = 240, h = 58 }: { marks: Marks; axis: readonly [number, number]; w?: number; h?: number }) {
  const [from, to] = axis
  const [lo, hi] = marks.scale
  const X = (at: number) => clamp((at - from) / Math.max(1, to - from), 0, 1) * (w - 2)
  const Y = (v: number) => h - 1.5 - (clamp((v - lo) / Math.max(1e-9, hi - lo), 0, 1) * (h - 4))
  const path = (points: readonly (readonly [number, number])[]): string => 'M' + points.map(([at, v]) => `${X(at).toFixed(1)},${Y(v).toFixed(1)}`).join(' L')
  const last = marks.line[marks.line.length - 1]
  return (
    <svg viewBox={`0 0 ${w} ${h}`} style={{ width: '100%', height: 'auto' }} aria-hidden="true">
      <line x1="0" x2={w} y1="0.5" y2="0.5" stroke="var(--line)" />
      <line x1="0" x2={w} y1={h - 0.5} y2={h - 0.5} stroke="var(--line-2)" />
      {marks.band && marks.band.length > 1 ? (
        <path
          d={`${path(marks.band.map(([at, , high]) => [at, high]))} ${marks.band
            .slice()
            .reverse()
            .map(([at, low]) => `L${X(at).toFixed(1)},${Y(low).toFixed(1)}`)
            .join(' ')} Z`}
          fill="var(--accent)"
          fillOpacity=".15"
        />
      ) : null}
      {marks.dashed && marks.dashed.length > 1 ? (
        <path d={path(marks.dashed)} fill="none" stroke="var(--muted)" strokeWidth="1.3" strokeDasharray="3 3" strokeLinejoin="round" vectorEffect="non-scaling-stroke" />
      ) : null}
      {marks.line.length > 1 ? (
        <path d={path(marks.line)} fill="none" stroke="var(--accent)" strokeWidth="1.6" strokeLinejoin="round" vectorEffect="non-scaling-stroke" />
      ) : null}
      {last ? <circle cx={X(last[0]).toFixed(1)} cy={Y(last[1]).toFixed(1)} r="2.4" fill="var(--accent)" /> : null}
    </svg>
  )
}

/**
 * How long a task took, rank by rank.
 *
 * Twelve buckets across the range, the slowest one marked: a task whose ranks finish together is
 * one column, and one straggler is a bar of its own far to the right, which is the whole point of
 * drawing it instead of listing twelve equal bars.
 */
export function Histo({ values, mark, labels, h = 84 }: { values: readonly number[]; mark?: number; labels: readonly [string, string]; h?: number }) {
  const w = 640
  const gap = 5
  const top = Math.max(...values, 1)
  const width = (w - gap * (values.length - 1)) / values.length
  return (
    <div>
      <svg viewBox={`0 0 ${w} ${h}`} style={{ display: 'block', width: '100%', height: 'auto' }} role="img" aria-label="how long the ranks took">
        {values.map((v, i) =>
          v ? (
            <rect
              key={i}
              x={(i * (width + gap)).toFixed(1)}
              y={(h - 1 - Math.max(3, (v / top) * (h - 6))).toFixed(1)}
              width={width.toFixed(1)}
              height={Math.max(3, (v / top) * (h - 6)).toFixed(1)}
              rx="3"
              fill={i === mark ? 'var(--warn)' : 'var(--accent)'}
              fillOpacity={i === mark ? 1 : 0.72}
            />
          ) : null,
        )}
        <line x1="0" x2={w} y1={h - 0.5} y2={h - 0.5} stroke="var(--line-2)" />
      </svg>
      <div className="axis">
        <span>{labels[0]}</span>
        <span>{labels[1]}</span>
      </div>
    </div>
  )
}

const REDUCED = typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches

/** The comb's staggered heartbeat, driven straight on the DOM as in the prototype. */
export function pulse(computeId: string): void {
  if (REDUCED) return
  const hxs = document.querySelectorAll<SVGGElement>(`[data-id="${computeId}"].hx`)
  hxs.forEach((g, i) =>
    setTimeout(
      () => {
        g.classList.add('beat')
        setTimeout(() => g.classList.remove('beat'), 800)
      },
      i * clamp(400 / Math.max(1, hxs.length), 3, 40),
    ),
  )
}
