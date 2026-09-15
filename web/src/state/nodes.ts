import type { Node } from '../api/client'
import type { NodeMetrics } from './model'
import type { CombNode } from '../ui/comb'

/** What the daemon last said about a node that is still coming up, keyed by node id. */
export type NodeProgress = { phase: string | null; completion: number | null; phases_done: number }

const EMPTY: NodeMetrics = { gpu: 0, vram: 0, cpu: 0, temp: 0, net: 0 }

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
): CombNode[] => nodes.map((n) => combNode(computeId, n, metrics, progress))

/** Every ready node's current value for one metric. */
export const valuesFor = (
  computeId: string,
  nodes: readonly Node[],
  metrics: Record<string, NodeMetrics>,
  metric: keyof NodeMetrics,
): number[] => nodes.filter((n) => n.state === 'ready').map((n) => (metrics[`${computeId}/${n.rank}`] ?? EMPTY)[metric])
