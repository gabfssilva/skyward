import type { ReactNode } from 'react'
import { Icon } from '../ui/icons'
import { useStore } from '../state/store'

/** The dimmed backdrop every sheet sits on; a click outside the sheet closes it. */
export function Scrim({ children, style }: { children: ReactNode; style?: React.CSSProperties }) {
  const close = useStore((s) => s.closeSheet)
  return (
    <div className="scrim" style={style} onMouseDown={(e) => { if (e.target === e.currentTarget) close() }}>
      {children}
    </div>
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
