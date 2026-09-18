import { useEffect, useRef, useState, type ReactNode } from 'react'
import { Icon, type IconName } from './icons'

/** One thing a page can do. The primary one is the page's verb; the rest are plain, and what does not fit is in the menu. */
export type Action = { label: string; icon?: IconName; danger?: boolean; disabled?: boolean; onClick: () => void }

/** Whether the frame is a phone's, which is what decides how many actions fit beside the title. */
export function useNarrow(): boolean {
  const [narrow, setNarrow] = useState(() => (typeof matchMedia === 'function' ? matchMedia('(max-width: 37.5em)').matches : false))
  useEffect(() => {
    if (typeof matchMedia !== 'function') return
    const watch = matchMedia('(max-width: 37.5em)')
    const answer = () => setNarrow(watch.matches)
    watch.addEventListener('change', answer)
    return () => watch.removeEventListener('change', answer)
  }, [])
  return narrow
}

/**
 * The actions of a page: one primary, up to two beside it, and the rest behind a menu.
 *
 * A phone keeps the primary and folds everything else in, so the row never wraps and the
 * destructive action is never the thing under a thumb.
 */
export function Actions({ primary, rest = [] }: { primary?: Action; rest?: readonly (Action | null)[] }) {
  const listed = rest.filter((a): a is Action => a !== null)
  const narrow = useNarrow()
  const [open, setOpen] = useState(false)
  const box = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const away = (e: MouseEvent) => {
      if (!box.current?.contains(e.target as globalThis.Node)) setOpen(false)
    }
    const escape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', away)
    document.addEventListener('keydown', escape)
    return () => {
      document.removeEventListener('mousedown', away)
      document.removeEventListener('keydown', escape)
    }
  }, [open])

  const beside = narrow ? [] : listed.filter((a) => !a.danger).slice(0, 2)
  const folded = listed.filter((a) => !beside.includes(a))
  if (!primary && !listed.length) return null

  return (
    <div className="acts" ref={box}>
      {primary ? (
        <button className="btn primary" aria-label={primary.label} title={narrow ? primary.label : undefined} disabled={primary.disabled} onClick={primary.onClick}>
          {primary.icon ? <Icon name={primary.icon} /> : null}
          {narrow ? null : primary.label}
        </button>
      ) : null}
      {beside.map((a) => (
        <button key={a.label} className="btn" disabled={a.disabled} onClick={a.onClick}>
          {a.label}
        </button>
      ))}
      {folded.length ? (
        <>
          <button className="btn icon" aria-expanded={open} aria-label="More actions" title="More actions" onClick={() => setOpen(!open)}>
            <Icon name="more" />
          </button>
          {open ? (
            <div className="menu" role="menu">
              {folded.map((a) => (
                <button
                  key={a.label}
                  role="menuitem"
                  className={a.danger ? 'danger' : undefined}
                  disabled={a.disabled}
                  onClick={() => {
                    setOpen(false)
                    a.onClick()
                  }}
                >
                  {a.icon ? <Icon name={a.icon} /> : null}
                  {a.label}
                </button>
              ))}
            </div>
          ) : null}
        </>
      ) : null}
    </div>
  )
}

/** What a page is of, one group per fact: the value in the ink, its context beside it in grey. */
export function Facts({ items }: { items: readonly (ReactNode | null | false)[] }) {
  const listed = items.filter(Boolean)
  if (!listed.length) return null
  return <div className="facts">{listed.map((fact, i) => <span key={i}>{fact}</span>)}</div>
}

/**
 * The same head on every route: where you are, what it is called, how it is doing, why it is not, what it is
 * made of, and what can be done to it.
 *
 * A page under a bar item carries no title — the bar already says Computes — so ``title`` is left out there
 * and the page opens on its summary instead. A breadcrumb is only for a level the bar cannot show, which is
 * the compute a node belongs to.
 */
export function PageHead({
  crumb,
  title,
  version,
  state,
  id,
  why,
  when,
  facts,
  primary,
  rest,
  children,
}: {
  crumb?: { label: string; onClick: () => void }
  title?: ReactNode
  version?: number | null
  state?: ReactNode
  id?: string
  /** what is wrong, and since when */
  why?: { message: string; since?: string } | null
  when?: string
  facts?: readonly (ReactNode | null | false)[]
  primary?: Action
  rest?: readonly (Action | null)[]
  /** what stands in for a title on a page that has none: the summary, or the tabs */
  children?: ReactNode
}) {
  return (
    <div className="head">
      {crumb ? (
        <div className="crumb">
          <button onClick={crumb.onClick}>{crumb.label}</button>
          <span aria-hidden="true">/</span>
        </div>
      ) : null}
      <div className="row">
        {title ? (
          <h1 className="title">
            {title}
            {version ? <span className="v">v{version}</span> : null}
          </h1>
        ) : null}
        {state}
        {id ? <span className="mono faint">{id}</span> : null}
        {when ? <span className="sub">{when}</span> : null}
        {children}
        <Actions primary={primary} rest={rest} />
      </div>
      {why ? (
        <div className="why">
          {why.message}
          {why.since ? <span>{why.since}</span> : null}
        </div>
      ) : null}
      {facts ? <Facts items={facts} /> : null}
    </div>
  )
}

/**
 * A card or a page split into views of the same thing, the count of each beside its name.
 *
 * The views are not the same height, and a page that is scrolled loses its place when the one below
 * shrinks. Whatever the new view is worth, the row itself is held where it was on the screen.
 */
export function Tabs<T extends string>({
  value,
  options,
  onChange,
  children,
}: {
  value: T
  options: readonly (readonly [T, string, number | null | undefined])[]
  onChange: (value: T) => void
  children?: ReactNode
}) {
  const row = useRef<HTMLDivElement>(null)
  const pick = (key: T) => {
    const was = row.current?.getBoundingClientRect().top
    onChange(key)
    if (was === undefined) return
    requestAnimationFrame(() => {
      const now = row.current?.getBoundingClientRect().top
      if (now !== undefined && Math.abs(now - was) > 1) window.scrollBy(0, now - was)
    })
  }
  return (
    <div className="tabs" role="tablist" ref={row}>
      {options.map(([key, label, count]) => (
        <button key={key} role="tab" aria-selected={value === key} onClick={() => pick(key)}>
          {label}
          {count === null || count === undefined ? null : <em>{count}</em>}
        </button>
      ))}
      {children}
    </div>
  )
}
