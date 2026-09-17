import { useEffect, useState } from 'react'
import { Navigate, Route, Routes, useLocation, useNavigate } from 'react-router-dom'
import * as fleet from './views/fleet'
import * as compute from './views/compute'
import * as node from './views/node'
import * as activity from './views/activity'
import * as tasks from './views/tasks'
import * as functions from './views/functions'
import * as market from './views/market'
import * as providers from './views/providers'
import { Sheets, openPalette } from './sheets'
import { Icon, type IconName } from './ui/icons'
import { Tip } from './ui/primitives'
import { pulse } from './ui/charts'
import { useStore } from './state/store'
import { money, rateOf } from './state/model'
import { api } from './api/client'
import { MOCK } from './api/mock'
import { applyStoredTheme, setTheme, type Theme } from './theme'
import './styles/index.css'

const VIEWS: readonly (readonly [string, string, IconName])[] = [
  ['/', 'Computes', 'fleet'],
  ['/tasks', 'Tasks', 'tasks'],
  ['/functions', 'Functions', 'functions'],
  ['/activity', 'Activity', 'activity'],
  ['/market', 'Market', 'market'],
  ['/providers', 'Providers', 'providers'],
]

const THEMES: readonly (readonly [Theme, IconName])[] = [
  ['auto', 'auto'],
  ['light', 'sun'],
  ['dark', 'moon'],
]

/** The view a path belongs to, so `/computes/:id` still lights `Computes`. */
const viewOf = (pathname: string): string => {
  const head = pathname.split('/').filter(Boolean)[0]
  return head && head !== 'computes' ? `/${head}` : '/'
}

/** What the whole account is running, in the bar, on every page but the home. */
function FleetTag() {
  const navigate = useNavigate()
  const computes = useStore((s) => s.computes)
  const nodesByCompute = useStore((s) => s.nodes)
  const rate = computes.reduce((sum, c) => sum + rateOf(nodesByCompute[c.id] ?? []), 0)
  const ready = computes.flatMap((c) => (nodesByCompute[c.id] ?? []).filter((n) => n.state === 'ready'))
  const gpus = computes.reduce((sum, c) => sum + (nodesByCompute[c.id] ?? []).filter((n) => n.state === 'ready').length * (c.spec.specs[0]?.accelerator_count ?? 1), 0)
  return (
    <button className="tag fleet" title="Computes" onClick={() => navigate('/')}>
      {money(rate, 0)}
      <small>/h</small>
      <span className="fleet-more">
        <i />
        {ready.length} nodes
        <i />
        {gpus} GPUs
      </span>
    </button>
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
      <div className="brand">
        <svg width="22" height="24" viewBox="0 0 17 19" aria-hidden="true" style={{ color: 'var(--accent)' }}>
          <polygon points="8.5,0.6 16.4,5.1 16.4,14.1 8.5,18.6 0.6,14.1 0.6,5.1" fill="currentColor" />
        </svg>
        <span>Skyward</span>
      </div>
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

/** The rail: big meters on the home, a compact crumb on a compute, node or task, nothing anywhere else. */
function RailRoutes() {
  return (
    <Routes>
      <Route path="/" element={<fleet.Rail />} />
      <Route path="/computes/:id" element={<compute.Rail />} />
      <Route path="/computes/:id/nodes/:rank" element={<node.Rail />} />
      <Route path="/tasks/:id" element={<tasks.TaskRail />} />
      <Route path="*" element={null} />
    </Routes>
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
      <Route path="/functions" element={<functions.Stage />} />
      <Route path="/market" element={<market.Stage />} />
      <Route path="/providers" element={<providers.Stage />} />
    </Routes>
  )
}

function InspectorRoutes() {
  return (
    <Routes>
      <Route path="/" element={<fleet.Inspector />} />
      <Route path="/computes/:id" element={<compute.Inspector />} />
      <Route path="/computes/:id/nodes/:rank" element={<node.Inspector />} />
      <Route path="/tasks" element={<tasks.Inspector />} />
      <Route path="/tasks/:id" element={<tasks.TaskInspector />} />
      <Route path="/functions" element={<functions.Inspector />} />
      <Route path="/market" element={<market.Inspector />} />
      <Route path="/providers" element={<providers.Inspector />} />
      <Route path="*" element={null} />
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
  const { pathname } = useLocation()
  return (
    <>
      <Bar />
      <div className={pathname === '/' ? 'rail' : 'rail compact'} id="rail">
        <RailRoutes />
      </div>
      <div className={pathname === '/activity' ? 'app wide' : 'app'}>
        <div className="stage" id="stage">
          <StageRoutes />
        </div>
        <aside className="inspector" id="inspector">
          <InspectorRoutes />
        </aside>
      </div>
      <Sheets />
      <Tip />
    </>
  )
}
