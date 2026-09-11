import { useEffect, useState } from 'react'
import { Route, Routes, useLocation, useNavigate } from 'react-router-dom'
import * as fleet from './views/fleet'
import * as computes from './views/computes'
import * as compute from './views/compute'
import * as tasks from './views/tasks'
import * as market from './views/market'
import * as providers from './views/providers'
import { Dock } from './dock'
import { Sheets, openPalette } from './sheets'
import { Icon, type IconName } from './ui/icons'
import { Tip } from './ui/primitives'
import { pulse } from './ui/charts'
import { useStore, pushSpend } from './state/store'
import { rateOf } from './state/model'
import { api } from './api/client'
import { MOCK } from './api/mock'
import { applyStoredTheme, setTheme, type Theme } from './theme'
import './theme.css'

const VIEWS: readonly (readonly [string, string, IconName])[] = [
  ['/', 'Fleet', 'fleet'],
  ['/computes', 'Computes', 'computes'],
  ['/tasks', 'Tasks', 'tasks'],
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
  return head ? `/${head}` : '/'
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
          <button key={path} role="tab" aria-selected={at === path} onClick={() => navigate(path)}>
            <Icon name={icon} />
            <span>{label}</span>
          </button>
        ))}
      </nav>
      <div className="row" style={{ marginLeft: 'auto', gap: 6 }}>
        {MOCK ? <span className="tag proto">example data</span> : null}
        <span className="tag" id="daemon-tag">
          <i className="dot ready" /> 127.0.0.1:17590{version ? ` · v${version}` : ''}
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
        <button className="iconbtn" title="Search  ⌘K" aria-label="Search" onClick={openPalette}>
          <Icon name="search" />
        </button>
      </div>
    </div>
  )
}

function RailRoutes() {
  return (
    <Routes>
      <Route path="/" element={<fleet.Rail />} />
      <Route path="/computes" element={<fleet.Rail />} />
      <Route path="/computes/:id" element={<compute.Rail />} />
      <Route path="/tasks" element={<fleet.Rail />} />
      <Route path="/tasks/:id" element={<tasks.TaskRail />} />
      <Route path="/market" element={<fleet.Rail />} />
      <Route path="/providers" element={<fleet.Rail />} />
    </Routes>
  )
}

function StageRoutes() {
  return (
    <Routes>
      <Route path="/" element={<fleet.Stage />} />
      <Route path="/computes" element={<computes.Stage />} />
      <Route path="/computes/:id" element={<compute.Stage />} />
      <Route path="/tasks" element={<tasks.Stage />} />
      <Route path="/tasks/:id" element={<tasks.TaskStage />} />
      <Route path="/market" element={<market.Stage />} />
      <Route path="/providers" element={<providers.Stage />} />
    </Routes>
  )
}

function InspectorRoutes() {
  return (
    <Routes>
      <Route path="/" element={<fleet.Inspector />} />
      <Route path="/computes" element={<computes.Inspector />} />
      <Route path="/computes/:id" element={<compute.Inspector />} />
      <Route path="/tasks" element={<tasks.Inspector />} />
      <Route path="/tasks/:id" element={<tasks.TaskInspector />} />
      <Route path="/market" element={<market.Inspector />} />
      <Route path="/providers" element={<providers.Inspector />} />
    </Routes>
  )
}

const REDUCED = typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches

/**
 * The prototype's `tick`, minus the parts the daemon now owns.
 *
 * Gauges and log lines arrive on the event stream, so all that is left here is
 * the fleet's spend series and the pulse that walks a busy compute's hexes.
 */
function useHeartbeat(): void {
  useEffect(() => {
    if (REDUCED) return
    let beat = 0
    const id = setInterval(() => {
      beat++
      const state = useStore.getState()
      pushSpend(state.computes.reduce((sum, c) => sum + rateOf(state.nodes[c.id] ?? []), 0))
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
      <div className="rail" id="rail">
        <RailRoutes />
      </div>
      <div className="app">
        <div className="stage" id="stage">
          <StageRoutes />
        </div>
        <aside className="inspector" id="inspector">
          <InspectorRoutes />
        </aside>
      </div>
      <Dock />
      <Sheets />
      <Tip />
    </>
  )
}
