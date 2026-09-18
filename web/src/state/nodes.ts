import type { Compute, Node } from '../api/client'
import { KIND, KIND_FILL, acceleratedOf, holdersOf, loadOf, money } from './model'
import type { Hex } from '../ui/comb'

/** One bootstrap phase of a node, in the state its last mark left it. */
export type PhaseMark = { name: string; state: 'started' | 'completed' | 'failed' }

/**
 * What the daemon last said about a node that is still coming up, keyed by node id.
 *
 * ``phases`` is the checklist in the order the machine reached it: which phases a
 * bootstrap runs depends on the image and the plugins, and only the node knows them.
 */
export type NodeProgress = { phase: string | null; completion: number | null; phases: readonly PhaseMark[] }

/** The newest reading of every metric one node reported, keyed ``${computeId}/${rank}``. */
export type Readings = Record<string, Record<string, number>>

/** A ready cell is the accent, deepened by how hard the machine is working; any other cell is its state's colour. */
export const toneOf = (state: string, load: number | null): string => {
  const kind = KIND(state)
  if (kind !== 'ready') return KIND_FILL[kind]
  return `color-mix(in oklab, var(--ok) ${Math.round(38 + Math.max(0, Math.min(100, load ?? 0)) * 0.62)}%, var(--panel))`
}

const tipOf = (n: Node, load: number | null, gauge: string, phase: string | null): string =>
  n.state === 'ready'
    ? `rank ${n.rank} · ${gauge} ${load === null ? '—' : `${Math.round(load)}%`} · ${n.address ?? '—'} · ${money(n.price_per_hour ?? 0)}/h`
    : `rank ${n.rank} · ${n.state}${n.last_error?.message ? ` — ${n.last_error.message}` : phase ? ` — ${phase}` : ''}`

/** One cell per rank, coloured by state and load; ``marked`` ranks are the ones the page names beside the drawing. */
export const nodeCells = (
  c: Compute,
  nodes: readonly Node[],
  readings: Readings,
  progress: Record<string, NodeProgress> = {},
  marked: ReadonlySet<number> = new Set(),
): Hex[] => {
  const accelerated = acceleratedOf(c)
  return holdersOf(nodes).map((n) => {
    const load = loadOf(readings[`${c.id}/${n.rank}`], accelerated)
    const kind = KIND(n.state)
    return {
      rank: n.rank,
      fill: toneOf(n.state, load),
      tip: tipOf(n, load, accelerated ? 'accelerator' : 'cpu', progress[n.id]?.phase ?? null),
      cls: kind === 'boot' ? 'boot' : undefined,
      marked: marked.has(n.rank),
    }
  })
}

/** Every ready node's newest load, by rank, ordered by whoever asks. */
export const loadByRank = (c: Compute, nodes: readonly Node[], readings: Readings): { rank: number; load: number }[] => {
  const accelerated = acceleratedOf(c)
  return nodes
    .filter((n) => n.state === 'ready')
    .map((n) => ({ rank: n.rank, load: loadOf(readings[`${c.id}/${n.rank}`], accelerated) }))
    .filter((x): x is { rank: number; load: number } => x.load !== null)
}
