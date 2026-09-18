import { useState, type ReactNode } from 'react'
import type { Node } from '../../api/client'
import { GAUGES, figure, holdersOf, lineValue, median } from '../../state/model'
import type { Gauge, Line } from '../../state/model'
import { WINDOWS, acrossNodes, axisOf, customMetric, customNames, newest, onNode, spanOver, useMetrics, useSpan } from '../../state/metrics'
import type { Feed, Marks, Window } from '../../state/metrics'
import type { Readings } from '../../state/nodes'
import { useStore } from '../../state/store'
import { Plot } from '../../ui/charts'

/** What one node reads for a line now: the live stream where there is one, the newest sample the daemon kept where there is not. */
const nowOf = (line: Line, computeId: string, n: Node, readings: Readings, feed: Feed | null): number | null => {
  const live = lineValue(line, readings[`${computeId}/${n.rank}`])
  if (live !== null) return live
  for (const name of line.names) {
    const kept = newest(feed, n.id, name)
    if (kept !== null) return kept * line.factor
  }
  return null
}

const valuesOf = (line: Line, computeId: string, nodes: readonly Node[], readings: Readings, feed: Feed | null): number[] =>
  nodes.map((n) => nowOf(line, computeId, n, readings, feed)).filter((v): v is number => v !== null)

/** A metric of somebody's own carries no unit and no scale, so it is printed the way a number reads: a loss to four places, a throughput whole. */
const plain = (v: number): string =>
  Math.abs(v) >= 1000 ? Math.round(v).toLocaleString('en-US') : v.toLocaleString('en-US', { maximumFractionDigits: 4 })

function Chart({ label, mono, value, unit, note, marks, axis }: { label: string; mono?: boolean; value: ReactNode; unit?: string; note?: string; marks: Marks; axis: readonly [number, number] }) {
  return (
    <div className="chart">
      <span className={mono ? 'l mono' : 'l'}>{label}</span>
      <span className="val">
        <span className="num">
          {value}
          {unit ? <small>{unit}</small> : null}
        </span>
        {note ? <span className="r">{note}</span> : null}
      </span>
      <div className="plot">
        <Plot marks={marks} axis={axis} />
      </div>
    </div>
  )
}

/**
 * Everything the nodes measure, over a window somebody picked, on one time axis.
 *
 * A compute's chart is the median across its nodes with the band from the lowest to the highest; a node's
 * chart is its own line with that median dashed behind it, which is how a machine slower than its peers
 * shows without a list of stragglers. The lines come off the daemon's history — ``/v1/computes/{id}/metrics``
 * — so a page that has just opened already has the window; the numbers beside them come off the live stream.
 */
export function Metrics({ computeId, name, nodes, created, over, node }: { computeId: string; name: string; nodes: readonly Node[]; created: number; over?: readonly [number, number | null]; node?: Node }) {
  const [window, setWindow] = useState<Window>('1h')
  const readings = useStore((s) => s.readings)
  const sliding = useSpan(window, created)
  const span = over ? spanOver(over[0], over[1]) : sliding
  const feed = useMetrics(computeId, span)
  const axis = axisOf(feed)
  /* a compute that has ended has no ready node: what it measured is what its machines measured while they were up */
  const ready = nodes.filter((n) => n.state === 'ready')
  const peers = ready.length ? ready : holdersOf(nodes)
  const reported = node ? [node] : peers
  const custom = customNames(feed)

  /** A gauge nothing reports is not drawn: a machine with no accelerator has no accelerator chart, rather than four flat lines. */
  const drawn = GAUGES.filter((g) => valuesOf(g.lines[0]!, computeId, reported, readings, feed).length > 0)
  if (!drawn.length && !custom.length) return null

  const chartOf = (g: Gauge) => {
    const mine = node ? nowOf(g.lines[0]!, computeId, node, readings, feed) : null
    const values = valuesOf(g.lines[0]!, computeId, reported, readings, feed)
    const all = node ? valuesOf(g.lines[0]!, computeId, peers, readings, feed) : values
    const value = node ? mine : values.length ? median(values) : null
    const total = g.total ? Math.max(0, ...valuesOf(g.total, computeId, reported, readings, feed)) : 0
    const second = g.lines[1] ? valuesOf(g.lines[1], computeId, reported, readings, feed) : []
    const note = total
      ? `of ${figure(g, total)} ${g.unit}`
      : second.length
        ? `${figure(g, median(second))} ${g.unit} out, dashed`
        : node
          ? all.length
            ? `median ${figure(g, median(all))}${g.unit}`
            : undefined
          : values.length > 1
            ? `${figure(g, Math.min(...values))} to ${figure(g, Math.max(...values))}${g.unit}`
            : undefined
    return (
      <Chart
        key={g.key}
        label={g.label}
        value={value === null ? '—' : figure(g, value)}
        unit={g.unit}
        note={note}
        marks={node ? onNode(feed, g, node.id) : acrossNodes(feed, g)}
        axis={axis}
      />
    )
  }

  const rows: readonly (readonly [string, ReactNode[]])[] = [
    ['Accelerator', drawn.filter((g) => g.group === 'Accelerator').map(chartOf)],
    ['Host', drawn.filter((g) => g.group === 'Host').map(chartOf)],
    [
      'Custom',
      custom.map((metric) => {
        const marks = customMetric(feed, metric, node?.id)
        const value = node ? newest(feed, node.id, metric) : median(nodes.map((n) => newest(feed, n.id, metric)).filter((v): v is number => v !== null))
        return <Chart key={metric} label={metric} mono value={value === null ? '—' : plain(value)} marks={marks} axis={axis} />
      }),
    ],
  ]

  return (
    <section className="card">
      <div className="mhead">
        <span className="h">Metrics</span>
        <span className="sub">{node ? `this node, with the median of ${name} dashed` : `median across ${reported.length} node${reported.length === 1 ? '' : 's'}, with the band from the lowest to the highest`}</span>
        {over ? (
          <span className="sub spread">while it ran</span>
        ) : (
          <div className="pick spread">
            {WINDOWS.map(([key, label]) => (
              <button key={key} aria-selected={window === key} onClick={() => setWindow(key)}>
                {label}
              </button>
            ))}
          </div>
        )}
      </div>
      {rows.map(([group, charts]) =>
        charts.length ? (
          <div className="mrow" key={group}>
            <span className="lbl">{group}</span>
            <div className="mcharts">{charts}</div>
          </div>
        ) : null,
      )}
    </section>
  )
}
