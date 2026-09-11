import { heat, last, norm, SCALE, UNIT, clamp, type MetricKey } from '../state/model'

export type BandPoint = { min: number; med: number; max: number }

export function Spark({ values, h = 44, fmt = (v: number) => v.toFixed(1) }: { values: readonly number[]; h?: number; fmt?: (v: number) => string }) {
  const w = 280
  if (values.length < 2) return <svg viewBox={`0 0 ${w} ${h}`} style={{ width: '100%', height: h, display: 'block' }} role="img" aria-label="trend" />
  const max = Math.max(...values) * 1.05 || 1
  const min = Math.min(...values) * 0.95
  const span = max - min || 1
  const X = (i: number) => (i / Math.max(1, values.length - 1)) * w
  const Y = (v: number) => h - 4 - ((v - min) / span) * (h - 10)
  const pts = values.map((v, i) => `${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(' L')
  return (
    <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" style={{ width: '100%', height: h, display: 'block' }} role="img" aria-label="trend">
      <path d={`M0,${h} L${pts} L${w},${h} Z`} fill="var(--accent)" opacity=".12" />
      <path d={`M${pts}`} fill="none" stroke="var(--accent)" strokeWidth="1.8" strokeLinejoin="round" />
      <circle cx={X(values.length - 1).toFixed(1)} cy={Y(last(values)).toFixed(1)} r="2.8" fill="var(--accent)" />
      {values.map((v, i) => (
        <rect
          key={i}
          x={(X(i) - w / values.length / 2).toFixed(1)}
          y="0"
          width={(w / values.length).toFixed(1)}
          height={h}
          fill="transparent"
          data-tip={`${fmt(v)} · ${values.length - i}m ago`}
        />
      ))}
    </svg>
  )
}

export function Band({ hist, h = 78 }: { hist: readonly BandPoint[]; h?: number }) {
  const w = 280
  if (!hist.length) return <svg viewBox={`0 0 ${w} ${h}`} style={{ width: '100%', height: h, display: 'block' }} />
  const X = (i: number) => (i / Math.max(1, hist.length - 1)) * w
  const Y = (v: number) => h - 10 - (v / 100) * (h - 18)
  const up = hist.map((d, i) => `${X(i).toFixed(1)},${Y(d.max).toFixed(1)}`).join(' L')
  const dn = hist
    .slice()
    .reverse()
    .map((d, i) => `${X(hist.length - 1 - i).toFixed(1)},${Y(d.min).toFixed(1)}`)
    .join(' L')
  const md = hist.map((d, i) => `${X(i).toFixed(1)},${Y(d.med).toFixed(1)}`).join(' L')
  return (
    <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" style={{ width: '100%', height: h, display: 'block' }} role="img" aria-label="cluster utilisation band">
      {[0, 50, 100].map((t) => (
        <line key={t} x1="0" x2={w} y1={Y(t).toFixed(1)} y2={Y(t).toFixed(1)} stroke="var(--line)" />
      ))}
      <path d={`M${up} L${dn} Z`} fill="var(--accent)" opacity=".18" />
      <path d={`M${md}`} fill="none" stroke="var(--accent)" strokeWidth="1.8" strokeLinejoin="round" />
      {hist.map((d, i) => (
        <rect
          key={i}
          x={(X(i) - w / hist.length / 2).toFixed(1)}
          y="0"
          width={(w / hist.length).toFixed(1)}
          height={h}
          fill="transparent"
          data-tip={`median ${Math.round(d.med)}% · slowest ${Math.round(d.min)}% · fastest ${Math.round(d.max)}%`}
        />
      ))}
    </svg>
  )
}

export function Histo({ values, metric, h = 62 }: { values: readonly number[]; metric: MetricKey; h?: number }) {
  const [lo, hi] = SCALE[metric]
  const step = (hi - lo) / 10
  const buckets = Array.from({ length: 10 }, (_, i) => values.filter((v) => v >= lo + i * step && (i === 9 ? v <= hi : v < lo + (i + 1) * step)).length)
  const top = Math.max(...buckets, 1)
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(10,1fr)', gap: 3, alignItems: 'end', height: h }}>
      {buckets.map((v, i) => (
        <div
          key={i}
          style={{ height: Math.max(3, (v / top) * h), background: heat(norm(metric, lo + (i + 0.5) * step)), borderRadius: 3 }}
          data-tip={`${v} node${v === 1 ? '' : 's'} · ${Math.round(lo + i * step)}–${Math.round(lo + (i + 1) * step)}${UNIT[metric]}`}
        />
      ))}
    </div>
  )
}

export type BarItem = { rank: number; value: number }

export function Bars({ items, metric, onPick }: { items: readonly BarItem[]; metric: MetricKey; onPick?: (rank: number) => void }) {
  return (
    <div className="bars">
      {items.map((x) => (
        <div key={x.rank} className="barrow" style={{ cursor: onPick ? 'pointer' : undefined }} onClick={() => onPick?.(x.rank)}>
          <span className="mono faint">rank {x.rank}</span>
          <span className="track">
            <i style={{ width: `${norm(metric, x.value)}%` }} />
          </span>
          <span className="mono right">
            {Math.round(x.value)}
            {UNIT[metric]}
          </span>
        </div>
      ))}
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
