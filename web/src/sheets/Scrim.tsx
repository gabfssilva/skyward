import { useLayoutEffect, useRef, type ReactNode } from 'react'
import { Icon } from '../ui/icons'
import { useStore } from '../state/store'

/**
 * How a sheet sits on the screen. A `full` sheet holds a form or a long read and takes the whole screen of a phone;
 * a `prompt` asks one question and stays a small box; the `palette` hangs from the top so the list under it has room.
 */
export type Layout = 'full' | 'prompt' | 'palette'

/**
 * The modal every sheet sits in.
 *
 * `showModal` puts it in the top layer and makes the page under it inert, so focus stays inside and Escape arrives as
 * `cancel`, which closes the sheet through the store like every other way out. A click on the dimmed margin closes
 * only a sheet that is `dismissible`: one holding a form would throw away what was typed.
 */
export function Scrim({ label, layout = 'full', dismissible = false, children }: { label: string; layout?: Layout; dismissible?: boolean; children: ReactNode }) {
  const close = useStore((s) => s.closeSheet)
  const ref = useRef<HTMLDialogElement>(null)

  useLayoutEffect(() => {
    if (ref.current && !ref.current.open) ref.current.showModal()
  }, [])

  return (
    <dialog
      ref={ref}
      className={`scrim ${layout}`}
      aria-label={label}
      onCancel={(e) => {
        e.preventDefault()
        close()
      }}
      onClose={close}
      onMouseDown={(e) => {
        if (dismissible && e.target === e.currentTarget) close()
      }}
    >
      {children}
    </dialog>
  )
}

export function CloseBtn({ style }: { style?: React.CSSProperties }) {
  const close = useStore((s) => s.closeSheet)
  return (
    <button className="iconbtn" aria-label="Close" style={style} onClick={close}>
      <Icon name="close" />
    </button>
  )
}
