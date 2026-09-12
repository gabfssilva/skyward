import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { functionName, nodesOf, useStore } from '../state/store'
import { clamp, money, rateOf } from '../state/model'
import { Icon } from '../ui/icons'
import { Scrim } from './Scrim'
import { VIEWS } from './catalog'
import { seedWizard } from './Wizard'

type Item = { label: string; hint: string; run: () => void }

export function Palette() {
  const navigate = useNavigate()
  const computes = useStore((s) => s.computes)
  const history = useStore((s) => s.history)
  const tasks = useStore((s) => s.tasks)
  const functions = useStore((s) => s.functions)
  const setUi = useStore((s) => s.setUi)
  const openSheet = useStore((s) => s.openSheet)
  const closeSheet = useStore((s) => s.closeSheet)
  const [q, setQ] = useState('')
  const [i, setI] = useState(0)
  const input = useRef<HTMLInputElement>(null)

  useEffect(() => {
    input.current?.focus()
  }, [])

  const items = useMemo<Item[]>(() => {
    const state = useStore.getState()
    const all: Item[] = [
      ...computes.map((c) => ({
        label: c.name ?? c.id,
        hint: `${nodesOf(state, c.id).length} nodes · ${money(rateOf(nodesOf(state, c.id)))}/h`,
        run: () => {
          closeSheet()
          navigate(`/computes/${c.id}`)
        },
      })),
      ...history.map((c) => ({
        label: c.name ?? c.id,
        hint: `compute · ${c.status.state}`,
        run: () => {
          closeSheet()
          navigate(`/computes/${c.id}`)
        },
      })),
      {
        label: 'New compute',
        hint: 'wizard',
        run: () => {
          seedWizard(undefined)
          openSheet({ kind: 'wizard' })
        },
      },
      ...VIEWS.map(([path, label]) => ({
        label,
        hint: 'go to',
        run: () => {
          closeSheet()
          navigate(path)
        },
      })),
      ...computes.flatMap((c) =>
        (tasks[c.id] ?? []).map((t) => ({
          label: `${functionName(state, t.function) ?? t.function.slice(0, 8)} · ${c.name ?? c.id}`,
          hint: `task · ${t.state}`,
          run: () => {
            closeSheet()
            navigate(`/tasks/${t.id}`)
          },
        })),
      ),
      ...computes.map((c) => ({
        label: `${c.name ?? c.id} · logs`,
        hint: 'in Activity',
        run: () => {
          closeSheet()
          setUi({ act: { ...state.act, kind: 'logs', compute: c.id, rank: 'all' } })
          navigate('/activity')
        },
      })),
      ...computes.map((c) => ({
        label: `${c.name ?? c.id} · shell`,
        hint: 'pty on rank 0',
        run: () => {
          closeSheet()
          setUi({ shell: true })
          navigate(`/computes/${c.id}/nodes/0`)
        },
      })),
    ]
    const needle = q.toLowerCase()
    return needle ? all.filter((x) => (x.label + x.hint).toLowerCase().includes(needle)) : all
  }, [computes, history, tasks, functions, q, navigate, setUi, openSheet, closeSheet])

  const idx = clamp(i, 0, Math.max(0, items.length - 1))

  const keys = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault()
      setI(clamp(idx + (e.key === 'ArrowDown' ? 1 : -1), 0, items.length - 1))
    } else if (e.key === 'Enter') {
      e.preventDefault()
      items[idx]?.run()
    }
  }

  return (
    <Scrim style={{ paddingTop: '13vh' }}>
      <div className="palette" role="dialog" aria-label="Search" onKeyDown={keys}>
        <div className="pin">
          <Icon name="search" />
          <input
            id="palette-input"
            ref={input}
            placeholder="Jump to a compute, its logs, a shell…"
            value={q}
            autoComplete="off"
            onChange={(e) => {
              setQ(e.target.value)
              setI(0)
            }}
          />
        </div>
        <ul>
          {items.length ? (
            items.map((item, n) => (
              <li key={`${item.label}/${item.hint}`} aria-selected={n === idx} onMouseEnter={() => setI(n)} onClick={() => item.run()}>
                <span>{item.label}</span>
                <span className="faint" style={{ marginLeft: 'auto', fontSize: 10.5 }}>
                  {item.hint}
                </span>
              </li>
            ))
          ) : (
            <li className="sub">Nothing matches.</li>
          )}
        </ul>
      </div>
    </Scrim>
  )
}
