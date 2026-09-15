import type { Node } from '../api/client'
import { holdersOf } from './model'
import type { NodeMetrics } from './model'
import type { CombNode } from '../ui/comb'

/** One bootstrap phase of a node, in the state its last mark left it. */
export type PhaseMark = { name: string; state: 'started' | 'completed' | 'failed' }

/**
 * What the daemon last said about a node that is still coming up, keyed by node id.
 *
 * ``phases`` is the checklist in the order the machine reached it: which phases a
 * bootstrap runs depends on the image and the plugins, and only the node knows them.
 */
export type NodeProgress = { phase: string | null; completion: number | null; phases: readonly PhaseMark[] }

const EMPTY: NodeMetrics = { gpu: 0, vram: 0, cpu: 0, temp: 0, rx: 0, tx: 0 }

/** The wire node, plus its live gauges and boot progress, in the shape the honeycomb draws. */
export const combNode = (
  computeId: string,
  n: Node,
  metrics: Record<string, NodeMetrics>,
  progress: Record<string, NodeProgress> = {},
): CombNode => {
  return {
    rank: n.rank,
    state: n.state,
    address: n.address ?? null,
    price: n.price_per_hour ?? 0,
    phase: progress[n.id]?.phase ?? null,
    error: n.last_error?.message ?? null,
    m: metrics[`${computeId}/${n.rank}`] ?? EMPTY,
  }
}

export const combNodes = (
  computeId: string,
  nodes: readonly Node[],
  metrics: Record<string, NodeMetrics>,
  progress: Record<string, NodeProgress> = {},
): CombNode[] => holdersOf(nodes).map((n) => combNode(computeId, n, metrics, progress))

/** Every ready node's current value for one metric. */
export const valuesFor = (
  computeId: string,
  nodes: readonly Node[],
  metrics: Record<string, NodeMetrics>,
  metric: keyof NodeMetrics,
): number[] => nodes.filter((n) => n.state === 'ready').map((n) => (metrics[`${computeId}/${n.rank}`] ?? EMPTY)[metric])
