import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import { useLocation } from 'react-router-dom'
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
    <span className={lap === null ? 'tick' : 'tick moving'} ref={box} style={lap === null ? undefined : { '--lap': `${lap}s` }}>
      <span>
        <span ref={copy}>{text}</span>
        {lap === null ? null : <span aria-hidden="true">{text}</span>}
      </span>
    </span>
  )
}

/** A table that scrolls sideways inside its card when the card is narrower, and a stop for the keyboard that scrolls it. */
export function TableScroll({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="tablebox" role="region" aria-label={label} tabIndex={0}>
      {children}
    </div>
  )
}

/** The element a `data-tip` belongs to, and its text. */
const tipAt = (at: Element | null): readonly [Element, string] | null => {
  const text = at?.getAttribute('data-tip')
  return at && text ? [at, text] : null
}

const targetOf = (e: Event): Element | null => (e.target instanceof Element ? e.target : null)

/**
 * The one floating tooltip, fed by any `data-tip` attribute on the page.
 *
 * A mouse or a pen shows it beside the pointer while it rests on the element, and keyboard focus shows it under the
 * element. A finger has no hover, so a tap shows it under what was tapped, and the next tap, a scroll, Escape or a
 * new page takes it away; only a tapped one goes with a scroll, or a log that follows its tail would keep taking a
 * resting pointer's away. It is a popover rather than a fixed box so that it is drawn over an open sheet, which sits
 * in the top layer too.
 */
export function Tip() {
  const ref = useRef<HTMLDivElement>(null)
  const { pathname } = useLocation()

  useEffect(() => {
    if (ref.current?.matches(':popover-open')) ref.current.hidePopover()
  }, [pathname])

  useEffect(() => {
    const tip = ref.current
    if (!tip) return
    let tapped = false
    const hide = () => {
      tapped = false
      if (tip.matches(':popover-open')) tip.hidePopover()
    }
    const show = ([at, text]: readonly [Element, string], pointer?: PointerEvent) => {
      tip.textContent = text
      if (!tip.matches(':popover-open')) tip.showPopover()
      const box = at.getBoundingClientRect()
      const [left, top] = pointer
        ? [pointer.clientX + 14, pointer.clientY - tip.offsetHeight - 12]
        : [box.left + box.width / 2 - tip.offsetWidth / 2, box.bottom + 8]
      tip.style.left = clamp(left, 8, innerWidth - tip.offsetWidth - 8) + 'px'
      tip.style.top = clamp(top, 8, innerHeight - tip.offsetHeight - 8) + 'px'
    }
    const move = (e: PointerEvent) => {
      if (e.pointerType === 'touch') return
      const hit = tipAt(targetOf(e)?.closest('[data-tip]') ?? null)
      if (hit) show(hit, e)
      else hide()
    }
    const tap = (e: PointerEvent) => {
      if (e.pointerType !== 'touch') return
      const hit = tipAt(targetOf(e)?.closest('[data-tip]') ?? null)
      if (!hit) return hide()
      show(hit)
      tapped = true
    }
    const scroll = () => {
      if (tapped) hide()
    }
    const focus = (e: FocusEvent) => {
      const at = targetOf(e)
      if (!at?.matches(':focus-visible')) return
      const hit = tipAt(at.closest('[data-tip]') ?? at.querySelector('[data-tip]'))
      if (hit) show(hit)
    }
    const escape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') hide()
    }
    document.addEventListener('pointermove', move)
    document.addEventListener('pointerup', tap)
    document.addEventListener('focusin', focus)
    document.addEventListener('focusout', hide)
    document.addEventListener('keydown', escape)
    document.addEventListener('scroll', scroll, { capture: true, passive: true })
    return () => {
      document.removeEventListener('pointermove', move)
      document.removeEventListener('pointerup', tap)
      document.removeEventListener('focusin', focus)
      document.removeEventListener('focusout', hide)
      document.removeEventListener('keydown', escape)
      document.removeEventListener('scroll', scroll, { capture: true })
    }
  }, [])

  return <div className="tip" id="tip" ref={ref} popover="manual" role="tooltip" />
}
