import { useState, type KeyboardEvent } from 'react'
import { api, type FunctionRef } from '../api/client'
import { useStore } from '../state/store'
import { Icon } from '../ui/icons'
import { Scrim, CloseBtn } from './Scrim'

const TEMPLATE = `import skyward as sky


def train(epochs: int = 1):
    return f"{epochs} epochs on {sky.instance_info().node}"
`

const INDENT = '    '

/**
 * The functions a module defines at its top level, in the order it defines them.
 *
 * The same rule the daemon parses for, by the one property a regular expression
 * can see it by: column zero. A nested function is indented, and is a detail of
 * the one around it rather than something a caller can name.
 */
const defines = (source: string): readonly string[] => [...source.matchAll(/^(?:async[ \t]+)?def[ \t]+([A-Za-z_]\w*)[ \t]*\(/gm)].map(([, name]) => name)

/** Tab indents and shift-tab outdents, as they do in every editor a person has used to write Python. */
function indent(event: KeyboardEvent<HTMLTextAreaElement>, replace: (source: string, caret: number) => void): void {
  if (event.key !== 'Tab') return
  event.preventDefault()
  const box = event.currentTarget
  const { value, selectionStart: at } = box
  if (!event.shiftKey) return replace(value.slice(0, at) + INDENT + value.slice(box.selectionEnd), at + INDENT.length)

  const line = value.lastIndexOf('\n', at - 1) + 1
  const width = value.slice(line, at).length - value.slice(line, at).replace(/^ {1,4}/, '').length
  if (width) replace(value.slice(0, line) + value.slice(line + width), at - width)
}

/**
 * A function written here rather than pickled from one that was already running.
 *
 * The name is not asked for, it is read: a module that defines one function is
 * calling that one, and a module that defines several offers the choice. Either
 * way the daemon parses the same source for the same name, so what is picked here
 * is a prediction of its answer and not a second opinion.
 */
export function Write({ from }: { from?: FunctionRef }) {
  const close = useStore((s) => s.closeSheet)
  const openSheet = useStore((s) => s.openSheet)
  const reload = useStore((s) => s.pageLibrary)
  const [source, setSource] = useState(from?.source ?? TEMPLATE)
  const [picked, setPicked] = useState<string | null>(from?.name ?? null)
  const [refusal, setRefusal] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const names = defines(source)
  const entry = picked !== null && names.includes(picked) ? picked : (names.at(-1) ?? null)

  const write = async (then: (fn: FunctionRef) => void) => {
    if (entry === null) return
    setBusy(true)
    setRefusal(null)
    try {
      const written = await api.writeFunction({ name: entry, source })
      await reload(true)
      then(written)
    } catch (error) {
      setRefusal(error instanceof Error ? error.message : String(error))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Scrim>
      <div className="sheet" style={{ width: 'min(760px,100%)' }} role="dialog" aria-label="Write a function">
        <div className="sheet-head">
          <b>{from ? `Edit ${from.name ?? 'function'}` : 'New function'}</b>
          <span className="sub">runs on a machine, not here</span>
          <CloseBtn style={{ marginLeft: 'auto' }} />
        </div>
        <div className="sheet-body" style={{ display: 'grid', gap: 10 }}>
          <div className="field">
            <label htmlFor="fn-source">Python</label>
            <textarea
              id="fn-source"
              className="code"
              spellCheck={false}
              rows={16}
              value={source}
              onChange={(e) => setSource(e.target.value)}
              onKeyDown={(e) =>
                indent(e, (next, caret) => {
                  setSource(next)
                  requestAnimationFrame(() => e.target instanceof HTMLTextAreaElement && e.target.setSelectionRange(caret, caret))
                })
              }
            />
          </div>
          <div className="row" style={{ gap: 8 }}>
            <span className="sub">Calls</span>
            {names.length > 1 ? (
              <select className="search" aria-label="entry point" value={entry ?? ''} onChange={(e) => setPicked(e.target.value)}>
                {names.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            ) : (
              <b className="mono">{entry ?? '—'}</b>
            )}
            <span className="sub" style={{ marginLeft: 'auto' }}>
              {names.length ? 'the imports it names must be on the machine' : 'define a function for a task to call'}
            </span>
          </div>
          {refusal ? (
            <div className="strip bad" style={{ alignItems: 'flex-start' }}>
              <Icon name="alert" />
              <span className="mono">{refusal}</span>
            </div>
          ) : null}
          <div className="row" style={{ gap: 6 }}>
            <button className="btn primary" disabled={busy || entry === null} onClick={() => void write(() => close())}>
              Save
            </button>
            <button
              className="btn"
              disabled={busy || entry === null}
              onClick={() => void write((written) => openSheet({ kind: 'run', lineage: written.lineage ?? undefined }))}
            >
              <Icon name="run" />
              Save and run
            </button>
          </div>
        </div>
      </div>
    </Scrim>
  )
}
