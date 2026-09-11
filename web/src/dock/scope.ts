import { useLocation } from 'react-router-dom'
import { useStore } from '../state/store'
import { taskOf } from '../state/model'

/** The compute the dock is scoped to: the open compute, the open task's compute, or none. */
export function useDockScope(): string | null {
  const path = useLocation().pathname
  const tasks = useStore((s) => s.tasks)
  const compute = path.match(/^\/computes\/([^/]+)/)
  if (compute) return compute[1] ?? null
  const task = path.match(/^\/tasks\/([^/]+)/)
  if (task && task[1]) return taskOf(tasks, task[1])?.computeId ?? null
  return null
}
