import { useState } from 'react'
import { useStore } from '../state/store'
import { Icon } from '../ui/icons'
import { Scrim, CloseBtn } from './Scrim'

export function Confirm({
  title,
  body,
  confirm,
  danger,
  onConfirm,
}: {
  title: string
  body: string
  confirm: string
  danger?: boolean
  onConfirm: () => void
}) {
  const closeSheet = useStore((s) => s.closeSheet)
  const [busy, setBusy] = useState(false)
  return (
    <Scrim label={title} layout="prompt" dismissible>
      <div className="sheet" style={{ width: 'min(420px,100%)' }}>
        <div className="sheet-head">
          <b>{title}</b>
          <CloseBtn style={{ marginLeft: 'auto' }} />
        </div>
        <div className="sheet-body">
          <div className="sub">{body}</div>
        </div>
        <div className="sheet-foot">
          <button className="btn" onClick={closeSheet}>
            Keep
          </button>
          <button
            className={danger === false ? 'btn primary' : 'btn danger'}
            style={{ marginLeft: 'auto' }}
            disabled={busy}
            onClick={() => {
              setBusy(true)
              onConfirm()
              closeSheet()
            }}
          >
            {danger === false ? null : <Icon name="trash" />}
            {confirm}
          </button>
        </div>
      </div>
    </Scrim>
  )
}
