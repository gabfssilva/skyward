import { useEffect, useState } from 'react'
import { Link, Navigate, Route, Routes, useLocation, useNavigate } from 'react-router-dom'
import * as fleet from './views/fleet'
import * as compute from './views/compute'
import * as node from './views/node'
import * as activity from './views/activity'
import * as tasks from './views/tasks'
import * as market from './views/market'
import { Sheets, openPalette } from './sheets'
import { Icon, type IconName } from './ui/icons'
import { Tip } from './ui/primitives'
import { pulse } from './ui/charts'
import { Logo } from './ui/comb'
import { useStore } from './state/store'
import { ago, money, rateOf } from './state/model'
import { described, useNotices } from './state/notices'
import { api } from './api/client'
import { MOCK } from './api/mock'
import { applyStoredTheme, setTheme, type Theme } from './theme'
import './styles/index.css'

/**
 * Four places to be: what is running, what it is being asked to do, what it has said, and what can be bought.
 *
 * Functions live inside Tasks — a function is what a task ran, and a library nobody has dispatched from is not a
 * place of its own — and provider accounts are a tab of the market, where their offers are.
 */
const VIEWS: readonly (readonly [string, string, IconName])[] = [
  ['/', 'Computes', 'fleet'],
  ['/tasks', 'Tasks', 'tasks'],
  ['/activity', 'Activity', 'activity'],
  ['/market', 'Market', 'market'],
]

const THEMES: readonly (readonly [Theme, IconName])[] = [
  ['auto', 'auto'],
  ['light', 'sun'],
  ['dark', 'moon'],
]

/** The view a path belongs to, so `/computes/:id` still lights `Computes` and an old `/providers` link still lights `Market`. */
const viewOf = (pathname: string): string => {
  const head = pathname.split('/').filter(Boolean)[0]
  switch (head) {
    case undefined:
    case 'computes':
      return '/'
    case 'functions':
      return '/tasks'
    case 'providers':
      return '/market'
    default:
      return `/${head}`
  }
}

/** What the whole account is burning, in the bar, on every page but the home. */
function FleetTag() {
  const navigate = useNavigate()
  const computes = useStore((s) => s.computes)
  const nodesByCompute = useStore((s) => s.nodes)
  const rate = computes.reduce((sum, c) => sum + rateOf(nodesByCompute[c.id] ?? []), 0)
  return (
    <button className="tag fleet" title="Computes" onClick={() => navigate('/')}>
      {money(rate, 0)}
      <small>/h</small>
    </button>
  )
}

/**
 * What failed lately and is not yet seen, behind a bell in the bar on every page.
 *
 * A notice opens what it is about: the task, for a task that failed, else the compute. Seen takes it off the
 * list at once; a failure that happens again is a notice again.
 */
function Notices() {
  const navigate = useNavigate()
  const { notices, see } = useNotices()
  const computes = useStore((s) => s.computes)
  const history = useStore((s) => s.history)
  const learn = useStore((s) => s.learn)
  const names = new Map([...computes, ...history].map((c) => [c.id, c.name ?? c.id]))
  const unknown = [...new Set(notices.map((n) => n.compute))].filter((id) => !names.has(id))
  useEffect(() => {
    if (unknown.length) void learn(unknown)
  }, [unknown.join(), learn])
  const open = (n: (typeof notices)[number]) => {
    document.getElementById('notices')?.hidePopover()
    navigate(n.kind.startsWith('task.') && n.subject ? `/tasks/${n.subject}` : `/computes/${n.compute}`)
  }
  return (
    <>
      <button className="iconbtn notices-btn" popoverTarget="notices" title="Notifications" aria-label={notices.length ? `Notifications, ${notices.length}` : 'Notifications'}>
        <Icon name="bell" />
        {notices.length ? <i className="count">{notices.length}</i> : null}
      </button>
      <div className="notices" id="notices" popover="auto">
        <div className="chead">
          <span className="h">Notifications</span>
          {notices.length ? (
            <button className="btn ghost sm spread" onClick={() => see(notices.flatMap((n) => n.ids))}>
              Mark all seen
            </button>
          ) : null}
        </div>
        {notices.length ? null : <div className="sub">Nothing failed in the last day that is not seen.</div>}
        {notices.map((n) => (
          <div className="notice" key={n.ids[0]}>
            <i className="sev" />
            <button className="what trunc" onClick={() => open(n)}>
              <b>{names.get(n.compute) ?? n.compute}</b> {described(n.kind)}
              {n.ids.length > 1 ? ` ×${n.ids.length}` : ''}
            </button>
            <span className="sub nowrap">{ago(n.at)}</span>
            <button className="btn sm" onClick={() => see(n.ids)}>
              Seen
            </button>
            {n.reason ? <div className="reason">{n.reason}</div> : null}
          </div>
        ))}
      </div>
    </>
  )
}

function Bar() {
  const { pathname } = useLocation()
  const navigate = useNavigate()
  const [theme, setThemeState] = useState<Theme>('auto')
  const [version, setVersion] = useState<string | null>(null)
  useEffect(() => setThemeState(applyStoredTheme()), [])
  useEffect(() => {
    void api
      .health()
      .then((h) => setVersion(h.version))
      .catch(() => setVersion(null))
  }, [])
  const at = viewOf(pathname)
  return (
    <div className="bar">
      <Link className="brand" to="/" aria-label="Skyward, home">
        <Logo />
        <span>Skyward</span>
      </Link>
      <nav id="nav" role="tablist">
        {VIEWS.map(([path, label, icon]) => (
          <button key={path} role="tab" aria-selected={at === path} aria-label={label} data-view={path} onClick={() => navigate(path)}>
            <Icon name={icon} />
            <span>{label}</span>
          </button>
        ))}
      </nav>
      <div className="bar-end">
        {pathname === '/' ? null : <FleetTag />}
        <Notices />
        <div className="bar-status" id="bar-status" popover="auto">
          {MOCK ? <span className="tag proto">example data</span> : null}
          <span className="tag" id="daemon-tag" title={version ? `v${version}` : undefined}>
            <i className="dot ready" /> <span className="tag-t">127.0.0.1:17590{version ? ` · v${version}` : ''}</span>
          </span>
          <div className="theme" id="theme">
            {THEMES.map(([key, icon]) => (
              <button
                key={key}
                className="iconbtn"
                aria-pressed={theme === key}
                title={key}
                onClick={() => {
                  setTheme(key)
                  setThemeState(key)
                }}
              >
                <Icon name={icon} />
              </button>
            ))}
          </div>
        </div>
        <button className="iconbtn bar-more" popoverTarget="bar-status" title="Daemon and theme" aria-label="Daemon and theme">
          <Icon name="more" />
        </button>
        <button className="iconbtn" title="Search  ⌘K" aria-label="Search" onClick={openPalette}>
          <Icon name="search" />
        </button>
      </div>
    </div>
  )
}

function StageRoutes() {
  return (
    <Routes>
      <Route path="/" element={<fleet.Stage />} />
      <Route path="/computes" element={<Navigate to="/" replace />} />
      <Route path="/computes/:id" element={<compute.Stage />} />
      <Route path="/computes/:id/nodes/:rank" element={<node.Stage />} />
      <Route path="/activity" element={<activity.Stage />} />
      <Route path="/tasks" element={<tasks.Stage />} />
      <Route path="/tasks/:id" element={<tasks.TaskStage />} />
      <Route path="/functions" element={<Navigate to="/tasks" replace />} />
      <Route path="/market" element={<market.Stage />} />
      <Route path="/market/accounts" element={<market.Stage accounts />} />
      <Route path="/providers" element={<Navigate to="/market/accounts" replace />} />
    </Routes>
  )
}

const REDUCED = typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches

/**
 * The prototype's `tick`, minus the parts the daemon now owns.
 *
 * Gauges, costs and log lines arrive on the event stream, so all that is left
 * here is the pulse that walks a busy compute's hexes.
 */
function useHeartbeat(): void {
  useEffect(() => {
    if (REDUCED) return
    let beat = 0
    const id = setInterval(() => {
      beat++
      const state = useStore.getState()
      if (beat % 4 === 0) {
        const busy = state.computes.find((c) => (state.tasks[c.id] ?? []).some((t) => t.state === 'running') && (state.nodes[c.id] ?? []).length > 1)
        if (busy) pulse(busy.id)
      }
    }, 2000)
    return () => clearInterval(id)
  }, [])
}

export default function App() {
  useHeartbeat()
  return (
    <>
      <Bar />
      <div className="app">
        <div className="stage" id="stage">
          <StageRoutes />
        </div>
      </div>
      <Sheets />
      <Tip />
    </>
  )
}
