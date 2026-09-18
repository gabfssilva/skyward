import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import { functionName, nodesOf, useStore } from '../state/store'
import { ago, clamp, money, ms, rateOf } from '../state/model'
import { Icon } from '../ui/icons'
import { Pill } from '../ui/primitives'
import { Scrim } from './Scrim'
import { VIEWS } from './catalog'
import { seedWizard } from './Wizard'

/** One thing the palette can take you to or do, under the heading it belongs to. */
type Item = { group: Group; key: string; label: ReactNode; text: string; hint: ReactNode; run: () => void }

type Group = 'Computes' | 'Tasks' | 'Actions' | 'Go to'

const GROUPS: readonly Group[] = ['Computes', 'Tasks', 'Actions', 'Go to']

/** Everything the palette can reach, grouped: what is running, what it is doing, what can be done, and where to go. */
export function Palette() {
  const navigate = useNavigate()
  const computes = useStore((s) => s.computes)
  const history = useStore((s) => s.history)
  const tasks = useStore((s) => s.tasks)
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
    const go = (path: string) => () => {
      closeSheet()
      navigate(path)
    }
    const all: Item[] = [
      ...[...computes, ...history].map((c) => ({
        group: 'Computes' as const,
        key: `compute/${c.id}`,
        label: (
          <>
            {c.name ?? c.id} <Pill state={c.status.state} />
          </>
        ),
        text: `${c.name ?? c.id} ${c.status.state}`,
        hint: `${nodesOf(state, c.id).length} nodes · ${money(rateOf(nodesOf(state, c.id)))}/h`,
        run: go(`/computes/${c.id}`),
      })),
      ...computes.flatMap((c) =>
        (tasks[c.id] ?? []).map((t) => {
          const name = t.function.name ?? functionName(state, t.function.sha256) ?? t.function.sha256.slice(0, 8)
          return {
            group: 'Tasks' as const,
            key: `task/${t.id}`,
            label: (
              <>
                {name} <Pill state={t.state} />
              </>
            ),
            text: `${name} ${t.state} ${c.name ?? c.id}`,
            hint: `${c.name ?? c.id} · ${ago(ms(t.submitted_at))}`,
            run: go(`/tasks/${t.id}`),
          }
        }),
      ),
      {
        group: 'Actions',
        key: 'action/new',
        label: 'New compute',
        text: 'new compute buy machines wizard',
        hint: 'wizard',
        run: () => {
          seedWizard(undefined)
          openSheet({ kind: 'wizard' })
        },
      },
      {
        group: 'Actions',
        key: 'action/write',
        label: 'New function',
        text: 'new function write code',
        hint: 'write one here',
        run: () => openSheet({ kind: 'write' }),
      },
      {
        group: 'Actions',
        key: 'action/run',
        label: 'Run a function',
        text: 'run a function dispatch task',
        hint: 'dispatch a task',
        run: () => openSheet({ kind: 'run' }),
      },
      ...computes.map((c) => ({
        group: 'Actions' as const,
        key: `shell/${c.id}`,
        label: `Shell on ${c.name ?? c.id}`,
        text: `shell terminal ${c.name ?? c.id}`,
        hint: 'a pty on rank 0',
        run: () => {
          closeSheet()
          setUi({ shell: true })
          navigate(`/computes/${c.id}/nodes/0`)
        },
      })),
      ...computes.map((c) => ({
        group: 'Actions' as const,
        key: `logs/${c.id}`,
        label: `Logs of ${c.name ?? c.id}`,
        text: `logs ${c.name ?? c.id}`,
        hint: 'in Activity',
        run: () => {
          closeSheet()
          setUi({ act: { ...state.act, kind: 'logs', compute: c.id, rank: 'all' } })
          navigate('/activity')
        },
      })),
      ...VIEWS.map(([path, label]) => ({ group: 'Go to' as const, key: `view/${path}`, label, text: label, hint: path, run: go(path) })),
    ]
    const needle = q.trim().toLowerCase()
    return needle ? all.filter((x) => x.text.toLowerCase().includes(needle)) : all
  }, [computes, history, tasks, q, navigate, setUi, openSheet, closeSheet])

  const ordered = useMemo(() => GROUPS.flatMap((group) => items.filter((x) => x.group === group)), [items])
  const idx = clamp(i, 0, Math.max(0, ordered.length - 1))

  const keys = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault()
      setI(clamp(idx + (e.key === 'ArrowDown' ? 1 : -1), 0, ordered.length - 1))
    } else if (e.key === 'Enter') {
      e.preventDefault()
      ordered[idx]?.run()
    }
  }

  return (
    <Scrim label="Search" layout="palette" dismissible>
      <div className="palette" onKeyDown={keys}>
        <div className="pin">
          <Icon name="search" />
          <input
            id="palette-input"
            ref={input}
            placeholder="Jump to a compute, a task, a shell…"
            value={q}
            autoComplete="off"
            onChange={(e) => {
              setQ(e.target.value)
              setI(0)
            }}
          />
        </div>
        <ul>
          {ordered.length ? (
            GROUPS.filter((group) => ordered.some((x) => x.group === group)).map((group) => (
              <li key={group} className="group">
                <span className="cap">{group}</span>
                <ul>
                  {ordered
                    .filter((x) => x.group === group)
                    .map((item) => (
                      <li key={item.key} aria-selected={ordered[idx] === item} onMouseEnter={() => setI(ordered.indexOf(item))} onClick={() => item.run()}>
                        <span className="row" style={{ gap: 8, minWidth: 0 }}>
                          {item.label}
                        </span>
                        <span className="faint spread nowrap" style={{ fontSize: 11 }}>
                          {item.hint}
                        </span>
                      </li>
                    ))}
                </ul>
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
