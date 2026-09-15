import { useEffect } from 'react'
import { useStore } from '../state/store'
import type { Offer } from '../api/client'
import { Wizard, seedWizard } from './Wizard'
import { Scale } from './Scale'
import { Ports } from './Ports'
import { AddProvider } from './AddProvider'
import { Confirm } from './Confirm'
import { Palette } from './Palette'

/** Open the wizard, optionally on an offer the market view picked. */
export const openWizard = (offer?: Offer): void => {
  seedWizard(offer)
  useStore.getState().openSheet({ kind: 'wizard' })
}

export const openScale = (computeId: string): void => useStore.getState().openSheet({ kind: 'scale', computeId })
export const openPorts = (computeId: string): void => useStore.getState().openSheet({ kind: 'ports', computeId })
export const openAddProvider = (provider?: string): void => useStore.getState().openSheet({ kind: 'addProvider', provider })
export const openPalette = (): void => useStore.getState().openSheet({ kind: 'palette' })
export const closeSheet = (): void => useStore.getState().closeSheet()

export const openConfirm = (confirm: {
  title: string
  body: string
  confirm: string
  danger?: boolean
  onConfirm: () => void
}): void => useStore.getState().openSheet({ kind: 'confirm', ...confirm })

/** Everything the overlay can hold, plus the two keys that open and close it. */
export function Sheets() {
  const sheet = useStore((s) => s.sheet)

  useEffect(() => {
    const keydown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        openPalette()
        return
      }
      if (e.key === 'Escape') {
        const state = useStore.getState()
        if (state.sheet) state.closeSheet()
      }
    }
    document.addEventListener('keydown', keydown)
    return () => document.removeEventListener('keydown', keydown)
  }, [])

  if (!sheet) return <div id="overlay" />
  return (
    <div id="overlay">
      {sheet.kind === 'wizard' ? <Wizard /> : null}
      {sheet.kind === 'scale' ? <Scale computeId={sheet.computeId} /> : null}
      {sheet.kind === 'ports' ? <Ports computeId={sheet.computeId} /> : null}
      {sheet.kind === 'addProvider' ? <AddProvider provider={sheet.provider} /> : null}
      {sheet.kind === 'confirm' ? (
        <Confirm title={sheet.title} body={sheet.body} confirm={sheet.confirm} danger={sheet.danger} onConfirm={sheet.onConfirm} />
      ) : null}
      {sheet.kind === 'palette' ? <Palette /> : null}
    </div>
  )
}
