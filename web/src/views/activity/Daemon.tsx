import { useEffect, useMemo, useRef, useState } from 'react'
import type { NavigateFunction } from 'react-router-dom'
import type { DaemonGroup, DaemonLine, DaemonSummary, LogLevel } from '../../api/client'
import { TINT, WINDOWS, useDaemonLog } from '../../state/daemonlog'
import type { DaemonFilters } from '../../state/daemonlog'
import { ago, clamp, clock } from '../../state/model'
import { useStore } from '../../state/store'
import { Icon } from '../../ui/icons'
import { Chip, Empty, Pick } from '../../ui/primitives'

const LEVEL_OPTIONS: readonly (readonly [LogLevel, string])[] = [
  ['DEBUG', 'all'],
  ['INFO', 'info'],
  ['WARNING', 'warnings'],
  ['ERROR', 'errors'],
]

const VIEWS: readonly (readonly [DaemonFilters['view'], string])[] = [
  ['grouped', 'Grouped'],
  ['lines', 'Lines'],
]

/**
 * What the daemon itself logged: counted on a chart, grouped by the call that logged it, and read line by line.
 *
 * Grouped is where a failure is found — thirty thousand lines of routine are one row, and the one call that kept
 * failing is a row of its own at the top. Lines is where it is read. Every id on a line narrows the page to it.
 */
export function Daemon() {
  const f = useStore((s) => s.daemon)
  const setUi = useStore((s) => s.setUi)
  const follow = useStore((s) => s.logFollow)
  const computes = useStore((s) => s.computes)
  const history = useStore((s) => s.history)
  const log = useDaemonLog(f)
  const names = useMemo(() => new Map([...computes, ...history].map((c) => [c.id, c.name ?? c.id])), [computes, history])
  const set = (patch: Partial<DaemonFilters>) => setUi({ daemon: { ...useStore.getState().daemon, ...patch } })

  const [draft, setDraft] = useState(f.q)
  useEffect(() => setDraft(f.q), [f.q])
  useEffect(() => {
    if (draft === f.q) return
    const typed = setTimeout(() => set({ q: draft }), 300)
    return () => clearTimeout(typed)
  }, [draft])

  if (log.missing)
    return (
      <section className="card">
        <Empty icon="activity" title="This daemon keeps no log">
          Only a daemon running on its own, as <span className="mono">sky server start</span> starts it, writes one.
        </Empty>
      </section>
    )

  const total = log.groups?.reduce((sum, g) => sum + g.count, 0) ?? 0
  const counted =
    f.view === 'grouped'
      ? log.groups
        ? `${log.groups.length} group${log.groups.length === 1 ? '' : 's'} · ${count(total)} line${total === 1 ? '' : 's'}`
        : ''
      : log.lines
        ? `${count(log.lines.length)} line${log.lines.length === 1 ? '' : 's'}${log.older ? ' loaded' : ''}`
        : ''

  return (
    <section className="card daemon">
      <Volume summary={log.summary} range={f.range} onRange={(range) => set({ range })} />

      <div className="row wrap" style={{ gap: 8 }}>
        <select
          className="search"
          aria-label="compute"
          style={{ minWidth: 150 }}
          value={f.compute}
          onChange={(e) => set({ compute: e.target.value, node: null })}
        >
          <option value="all">Every compute</option>
          {computes.map((c) => (
            <option key={c.id} value={c.id}>
              {c.name ?? c.id}
            </option>
          ))}
          {history.map((c) => (
            <option key={c.id} value={c.id}>
              {c.name ?? c.id} · ended
            </option>
          ))}
          {f.compute !== 'all' && !names.has(f.compute) ? <option value={f.compute}>{f.compute}</option> : null}
        </select>
        <Pick value={f.level} options={LEVEL_OPTIONS} onChange={(level) => set({ level })} />
        <Pick value={f.window} options={WINDOWS} onChange={(window) => set({ window, range: null })} />
        <input
          className="search"
          placeholder="filter messages and exceptions"
          aria-label="search"
          value={draft}
          autoComplete="off"
          onChange={(e) => setDraft(e.target.value)}
        />
        {f.view === 'lines' ? (
          <Chip pressed={follow} onClick={() => setUi({ logFollow: !follow })}>
            {follow ? 'following' : 'paused'}
          </Chip>
        ) : null}
      </div>

      <Components counts={log.summary?.components ?? {}} chosen={f.components} onToggle={(name) => set({ components: toggled(f.components, name) })} />
      <Narrowed f={f} names={names} set={set} />

      <div className="row wrap" style={{ gap: 8 }}>
        <Pick value={f.view} options={VIEWS} onChange={(view) => set({ view })} />
        <span className="sub spread">{counted}</span>
      </div>
      {log.error ? <span className="err-line">{log.error}</span> : null}

      {f.view === 'grouped' ? (
        <Groups
          groups={log.groups}
          names={names}
          onOpen={(group) => set({ group, view: 'lines' })}
          onHide={(group) => set({ hidden: [...f.hidden, group], group: f.group === group ? null : f.group })}
        />
      ) : (
        <Lines
          lines={log.lines}
          names={names}
          follow={follow}
          older={log.older}
          loadingOlder={log.loadingOlder}
          onOlder={log.loadOlder}
          onCompute={(compute) => set({ compute, node: null })}
          onNode={(node) => set({ node })}
        />
      )}
    </section>
  )
}

/** Open the daemon's log on one compute, or on one of its nodes: what a compute's page and a node's page link to. */
export function openDaemonLog(navigate: NavigateFunction, compute: string, node: string | null = null): void {
  const { act, daemon, setUi } = useStore.getState()
  setUi({ act: { ...act, kind: 'daemon' }, daemon: { ...daemon, compute, node, group: null, range: null, components: [] } })
  navigate('/activity')
}

/** How many lines the daemon logged on each step of the window, by level; dragging across it narrows the page to a stretch. */
function Volume({
  summary,
  range,
  onRange,
}: {
  summary: DaemonSummary | null
  range: readonly [number, number] | null
  onRange: (range: readonly [number, number] | null) => void
}) {
  const box = useRef<SVGSVGElement>(null)
  const [drag, setDrag] = useState<{ from: number; to: number } | null>(null)
  const volume = summary?.volume
  const steps = volume?.debug.length ?? 60
  const since = volume?.since ?? 0
  const step = volume?.step ?? 60_000
  const width = steps * step
  const totals = volume ? volume.debug.map((n, i) => n + volume.info[i]! + volume.warning[i]! + volume.error[i]!) : []
  const peak = Math.max(1, ...totals)
  const column = W / steps

  const at = (clientX: number): number => {
    const rect = box.current?.getBoundingClientRect()
    return rect ? since + clamp((clientX - rect.left) / rect.width, 0, 1) * width : since
  }
  const x = (t: number): number => clamp(((t - since) / width) * W, 0, W)
  const shown = drag ? ([Math.min(drag.from, drag.to), Math.max(drag.from, drag.to)] as const) : range

  return (
    <div className="dvol">
      <div className="row wrap" style={{ gap: 12 }}>
        <span className="cap">lines per {per(step)}</span>
        <span className="sub">{volume ? `peak ${count(peak)}` : ''}</span>
        <span className="dvol-legend spread" aria-hidden="true">
          {(['debug', 'info', 'warn', 'err'] as const).map((tint) => (
            <span key={tint}>
              <i className={`lv-${tint}`} />
              {tint === 'warn' ? 'warning' : tint === 'err' ? 'error' : tint}
            </span>
          ))}
        </span>
      </div>
      <svg
        ref={box}
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        role="img"
        aria-label="lines the daemon logged over the window, by level"
        onPointerDown={(e) => {
          if (!volume) return
          const t = at(e.clientX)
          setDrag({ from: t, to: t })
          e.currentTarget.setPointerCapture(e.pointerId)
        }}
        onPointerMove={(e) => {
          if (drag) setDrag({ ...drag, to: at(e.clientX) })
        }}
        onPointerUp={(e) => {
          if (!drag) return
          const t = at(e.clientX)
          const from = Math.min(drag.from, t)
          const to = Math.max(drag.from, t)
          const first = since + Math.floor((from - since) / step) * step
          setDrag(null)
          onRange(to - from < step ? [first, first + step] : [from, to])
        }}
      >
        {volume
          ? totals.map((_, i) => {
              let y = H
              const stacked = (['debug', 'info', 'warning', 'error'] as const).map((level) => {
                const n = volume[level][i]!
                if (!n) return null
                const h = Math.max(1, (n / peak) * (H - TOP))
                y -= h
                return <rect key={level} className={`lv-${BAR[level]}`} x={i * column + 1} y={y} width={Math.max(1, column - 2)} height={h} />
              })
              return (
                <g key={i}>
                  {stacked}
                  {volume.error[i] ? <rect className="lv-err" x={i * column + 1} y={0} width={Math.max(1, column - 2)} height={3} /> : null}
                </g>
              )
            })
          : null}
        {shown ? (
          <>
            <rect className="dvol-shade" x={0} y={0} width={x(shown[0])} height={H} />
            <rect className="dvol-shade" x={x(shown[1])} y={0} width={W - x(shown[1])} height={H} />
          </>
        ) : null}
      </svg>
      <div className="axis">
        <span>−{spanOf(width)}</span>
        <span>−{spanOf(width / 2)}</span>
        <span>now</span>
      </div>
    </div>
  )
}

function Components({ counts, chosen, onToggle }: { counts: Readonly<Record<string, number>>; chosen: readonly string[]; onToggle: (name: string) => void }) {
  const names = [...new Set([...Object.keys(counts), ...chosen])].sort((a, b) => (counts[b] ?? 0) - (counts[a] ?? 0))
  if (!names.length) return null
  return (
    <div className="chips" aria-label="components">
      {names.map((name) => (
        <Chip key={name} pressed={chosen.includes(name)} onClick={() => onToggle(name)}>
          {name} <em>{count(counts[name] ?? 0)}</em>
        </Chip>
      ))}
    </div>
  )
}

/** The filters that are not a control of their own — a node, a group, a stretch of the chart, the groups hidden — each one a chip that takes it away. */
function Narrowed({ f, names, set }: { f: DaemonFilters; names: ReadonlyMap<string, string>; set: (patch: Partial<DaemonFilters>) => void }) {
  const chips: (readonly [string, string, string, () => void])[] = [
    ...(f.node ? [['node', 'node', f.node, () => set({ node: null })] as const] : []),
    ...(f.group ? [['group', 'group', readable(f.group), () => set({ group: null })] as const] : []),
    ...(f.range ? [['range', 'window', `${clock(f.range[0])}–${clock(f.range[1])}`, () => set({ range: null })] as const] : []),
    ...f.hidden.map((group) => [`hidden:${group}`, 'hidden', readable(group), () => set({ hidden: f.hidden.filter((g) => g !== group) })] as const),
  ]
  if (!chips.length) return null
  return (
    <div className="chips">
      {chips.map(([key, label, value, clear]) => (
        <button key={key} className="chip narrowed" title="Remove this filter" onClick={clear}>
          <span>{label}</span>
          {label === 'node' ? short(value) : (names.get(value) ?? value)}
          <Icon name="close" />
        </button>
      ))}
    </div>
  )
}

function Groups({
  groups,
  names,
  onOpen,
  onHide,
}: {
  groups: readonly DaemonGroup[] | null
  names: ReadonlyMap<string, string>
  onOpen: (group: string) => void
  onHide: (group: string) => void
}) {
  if (groups === null) return <div className="dlist sub">Reading the log…</div>
  if (!groups.length) return <div className="dlist sub">No line matches.</div>
  return (
    <div className="dlist">
      {groups.map((g) => {
        const compute = g.latest.compute
        const where = g.computes > 1 ? `${g.computes} computes` : compute ? (names.get(compute) ?? short(compute)) : null
        const failed = g.latest.exception
        return (
          <div className={`dgroup ${TINT[g.level]}`} key={g.key}>
            <i className="sev" />
            <button className="dg-main" title="Show these lines" onClick={() => onOpen(g.key)}>
              <span className="dg-top">
                <span className="mono">{g.site}</span>
                {g.exception ? <span className="xt">{tail(g.exception)}</span> : null}
                <span className="sub">
                  {g.component ?? g.latest.logger}
                  {where ? ` · ${where}` : ''}
                </span>
              </span>
              <span className="dg-msg">
                {g.latest.message}
                {failed ? ` — ${failed.cause ?? `${tail(failed.type)}: ${failed.message}`}` : ''}
              </span>
            </button>
            <Spark series={g.series} />
            <span className="dg-count">
              {count(g.count)}
              <small>{g.count === 1 ? 'line' : 'lines'}</small>
            </span>
            <span className="dg-when">{ago(Date.parse(g.last))}</span>
            <button className="iconbtn dg-hide" title="Hide this group" aria-label="Hide this group" onClick={() => onHide(g.key)}>
              <Icon name="close" />
            </button>
          </div>
        )
      })}
    </div>
  )
}

function Spark({ series }: { series: readonly number[] }) {
  const w = 112
  const h = 26
  const top = Math.max(1, ...series)
  const points = series.map((n, i) => [(i / Math.max(1, series.length - 1)) * w, h - 2 - (n / top) * (h - 4)] as const)
  const line = points.map(([px, py], i) => `${i ? 'L' : 'M'}${px.toFixed(1)} ${py.toFixed(1)}`).join('')
  return (
    <svg className="spark" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" aria-hidden="true">
      <path className="a" d={`${line}L${w} ${h}L0 ${h}Z`} />
      <path className="l" d={line} />
    </svg>
  )
}

function Lines({
  lines,
  names,
  follow,
  older,
  loadingOlder,
  onOlder,
  onCompute,
  onNode,
}: {
  lines: readonly DaemonLine[] | null
  names: ReadonlyMap<string, string>
  follow: boolean
  older: boolean
  loadingOlder: boolean
  onOlder: () => void
  onCompute: (compute: string) => void
  onNode: (node: string) => void
}) {
  const box = useRef<HTMLDivElement>(null)
  const [open, setOpen] = useState<ReadonlySet<number>>(() => new Set())
  const newest = lines?.[lines.length - 1]?.sequence
  useEffect(() => {
    if (follow && box.current) box.current.scrollTop = box.current.scrollHeight
  }, [newest, follow])

  const toggle = (sequence: number) =>
    setOpen((held) => {
      const next = new Set(held)
      if (!next.delete(sequence)) next.add(sequence)
      return next
    })

  return (
    <div className="dlist" ref={box}>
      {lines === null ? (
        <span className="sub">Reading the log…</span>
      ) : (
        <>
          {older ? (
            <button className="btn sm" style={{ margin: '8px 0' }} disabled={loadingOlder} onClick={onOlder}>
              Load older lines
            </button>
          ) : null}
          {lines.length ? (
            lines.map((line) => (
              <Line
                key={line.sequence}
                line={line}
                names={names}
                open={open.has(line.sequence)}
                onToggle={() => toggle(line.sequence)}
                onCompute={onCompute}
                onNode={onNode}
              />
            ))
          ) : (
            <span className="sub">No line matches.</span>
          )}
        </>
      )}
    </div>
  )
}

function Line({
  line,
  names,
  open,
  onToggle,
  onCompute,
  onNode,
}: {
  line: DaemonLine
  names: ReadonlyMap<string, string>
  open: boolean
  onToggle: () => void
  onCompute: (compute: string) => void
  onNode: (node: string) => void
}) {
  const { compute, node, exception } = line
  const at = Date.parse(line.at)
  return (
    <div className={`dline ${TINT[line.level]}`}>
      <span className="t">
        {clock(at)}
        <small>.{String(at % 1000).padStart(3, '0')}</small>
      </span>
      <span className="lv">{line.level}</span>
      <span className="cmp" title={line.site}>
        {line.component ?? line.logger}
      </span>
      <div className="m">
        <span className="txt">{line.message}</span>
        {compute ? (
          <button className="fld" title={`Only ${compute}`} onClick={() => onCompute(compute)}>
            {names.get(compute) ?? short(compute)}
          </button>
        ) : null}
        {node ? (
          <button className="fld" title={`Only ${node}`} onClick={() => onNode(node)}>
            {short(node)}
          </button>
        ) : null}
        {Object.entries(line.fields).map(([key, value]) => (
          <span className="fld" key={key}>
            {key}={value}
          </span>
        ))}
        {exception ? (
          <>
            <button className="exc" aria-expanded={open} onClick={onToggle}>
              {open ? '▾' : '▸'} <b>{tail(exception.type)}</b>: {exception.message}
              {exception.cause ? <span className="cause">↳ {exception.cause}</span> : null}
            </button>
            {open ? <pre className="tb">{exception.traceback}</pre> : null}
          </>
        ) : null}
      </div>
    </div>
  )
}

const W = 600
const H = 76
const TOP = 7
const BAR = { debug: 'debug', info: 'info', warning: 'warn', error: 'err' } as const

const count = (n: number): string => n.toLocaleString('en-US')
const short = (id: string): string => (id.length > 12 ? `${id.slice(0, 12)}…` : id)
const tail = (type: string): string => type.split('.').pop() ?? type
const readable = (group: string): string => group.replace('|', ' · ')
const toggled = (held: readonly string[], name: string): string[] => (held.includes(name) ? held.filter((n) => n !== name) : [...held, name])
const per = (step: number): string => (step === 60_000 ? 'minute' : step < 60_000 ? `${Math.round(step / 1000)} s` : `${Math.round(step / 60_000)} min`)
const spanOf = (ms: number): string => (ms >= 7.2e6 ? `${+(ms / 3.6e6).toFixed(1)}h` : `${+(ms / 6e4).toFixed(1)}m`)
