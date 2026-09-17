import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react'
import { Icon, type IconName } from './icons'
import { clamp, legendOf, tally } from '../state/model'
import { useFunctionLabel } from '../state/store'

export function Btn({
  children,
  icon,
  variant,
  size,
  disabled,
  onClick,
}: {
  children?: ReactNode
  icon?: IconName
  variant?: 'primary' | 'ghost' | 'danger'
  size?: 'sm'
  disabled?: boolean
  onClick?: () => void
}) {
  return (
    <button className={['btn', variant, size].filter(Boolean).join(' ')} disabled={disabled} onClick={onClick}>
      {icon ? <Icon name={icon} /> : null}
      {children}
    </button>
  )
}

export function IconBtn({
  icon,
  title,
  pressed,
  onClick,
}: {
  icon: IconName
  title?: string
  pressed?: boolean
  onClick?: () => void
}) {
  return (
    <button className="iconbtn" title={title} aria-label={title} aria-pressed={pressed} onClick={onClick}>
      <Icon name={icon} />
    </button>
  )
}

export function Pill({ state }: { state: string }) {
  return (
    <span className={`state ${state}`}>
      <i className={`dot ${state}`} />
      {state.replace(/_/g, ' ')}
    </span>
  )
}

export function Chip({ children, pressed, onClick }: { children: ReactNode; pressed?: boolean; onClick?: () => void }) {
  return (
    <button className="chip" aria-pressed={pressed} onClick={onClick}>
      {children}
    </button>
  )
}

export function Pick<T extends string>({
  value,
  options,
  onChange,
}: {
  value: T
  options: readonly (readonly [T, ReactNode])[]
  onChange: (value: T) => void
}) {
  return (
    <div className="pick">
      {options.map(([key, label]) => (
        <button key={key} aria-selected={value === key} onClick={() => onChange(key)}>
          {label}
        </button>
      ))}
    </div>
  )
}

export function Card({ children, tight }: { children: ReactNode; tight?: boolean }) {
  return <div className={tight ? 'card tight' : 'card'}>{children}</div>
}

export function Kv({ k, v }: { k: ReactNode; v: ReactNode }) {
  return (
    <div className="kv">
      <span className="sub">{k}</span>
      <span>{v}</span>
    </div>
  )
}

/** A task's function: its name and the version the task ran, or the head of the sha the task actually carries. */
export function Fn({ sha, size, weight }: { sha: string; size?: number; weight?: number }) {
  const fn = useFunctionLabel(sha)
  return (
    <b className={fn.mono ? 'mono' : undefined} style={{ fontSize: size, fontWeight: weight ?? 600 }}>
      {fn.text}
      {fn.version ? <span className="faint mono" style={{ fontWeight: 400, fontSize: '0.85em' }}> v{fn.version}</span> : null}
    </b>
  )
}

export function Legend({ states }: { states: readonly string[] }) {
  const t = tally(states)
  const legend = legendOf(states)
  return (
    <div className="legend">
      {legend.filter(([k]) => t[k]).map(([k, label, token]) => (
        <span key={k}>
          <i style={{ background: `var(${token})` }} />
          <b>{t[k]}</b> {label}
        </span>
      ))}
    </div>
  )
}

export function Empty({ icon, title, children }: { icon: IconName; title: string; children?: ReactNode }) {
  return (
    <div className="empty">
      <Icon name={icon} />
      <b>{title}</b>
      {children ? <span>{children}</span> : null}
    </div>
  )
}

/**
 * A text longer than its box scrolls on a loop, which is what says there is more of it. The loop runs over two
 * copies, so moving by one copy and its gap is one lap, at about the same speed whatever the length. The box and the text
 * are watched rather than measured once: a box that narrows with the window, or a font that arrives late, changes
 * whether the text fits.
 */
export function Tick({ text }: { text: string }) {
  const box = useRef<HTMLSpanElement>(null)
  const copy = useRef<HTMLSpanElement>(null)
  const [lap, setLap] = useState<number | null>(null)

  useLayoutEffect(() => {
    const outer = box.current
    const inner = copy.current
    if (!outer || !inner) return
    const fit = () => setLap(inner.offsetWidth > outer.clientWidth ? inner.offsetWidth / 30 : null)
    const watch = new ResizeObserver(fit)
    watch.observe(outer)
    watch.observe(inner)
    return () => watch.disconnect()
  }, [])

  return (
    <span className={lap === null ? 'tick' : 'tick moving'} ref={box} style={lap === null ? undefined : ({ '--lap': `${lap}s` } as CSSProperties)}>
      <span>
        <span ref={copy}>{text}</span>
        {lap === null ? null : <span aria-hidden="true">{text}</span>}
      </span>
    </span>
  )
}

/** The one floating tooltip, fed by any `data-tip` attribute on the page. */
export function Tip() {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const move = (e: MouseEvent) => {
      const tip = ref.current
      if (!tip) return
      const target = e.target instanceof Element ? e.target.closest('[data-tip]') : null
      const text = target instanceof HTMLElement || target instanceof SVGElement ? target.getAttribute('data-tip') : null
      if (!text) {
        tip.classList.remove('on')
        return
      }
      tip.textContent = text
      tip.classList.add('on')
      tip.style.left = clamp(e.clientX + 14, 8, innerWidth - tip.offsetWidth - 8) + 'px'
      tip.style.top = clamp(e.clientY - tip.offsetHeight - 12, 8, innerHeight - tip.offsetHeight - 8) + 'px'
    }
    document.addEventListener('mousemove', move)
    return () => document.removeEventListener('mousemove', move)
  }, [])
  return <div className="tip" id="tip" ref={ref} />
}
