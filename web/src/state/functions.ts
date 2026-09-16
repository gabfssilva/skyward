import { useEffect, useState } from 'react'
import { api, type FunctionRef } from '../api/client'

/** One version of a function: the uploads that share its code, newest first, and the one that stands for them. */
export type Version = { version: number; uploads: FunctionRef[]; newest: FunctionRef }

/**
 * A function's uploads grouped into its versions, highest first.
 *
 * Uploads of one version share their code and differ in what a run captured or
 * where the function sat in its file, so the newest of them is the one to run.
 */
export const versionsOf = (uploads: readonly FunctionRef[]): Version[] => {
  const by = new Map<number, FunctionRef[]>()
  for (const upload of uploads) by.set(upload.version, [...(by.get(upload.version) ?? []), upload])
  return [...by.entries()]
    .sort(([a], [b]) => b - a)
    .map(([version, all]) => {
      const sorted = [...all].sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at))
      return { version, uploads: sorted, newest: sorted[0]! }
    })
}

/** Where a function was written: the file's name, or the console for one written there. */
export const whereOf = (fn: FunctionRef): string => (fn.source ? 'console' : (fn.origin?.split('/').at(-1) ?? 'unknown'))

/**
 * Every upload of one function, read when the lineage is first asked for.
 *
 * A function's history is only wanted while somebody is looking at it, so it is
 * kept by whoever looks rather than in the store every view shares. It is the
 * newest uploads that are read, so a function uploaded once per job for years
 * still shows every recent version and loses only the oldest.
 */
export function useLineage(lineage: string | null | undefined): { uploads: FunctionRef[]; loading: boolean } {
  const [state, setState] = useState<{ key: string | null; uploads: FunctionRef[] }>({ key: null, uploads: [] })
  useEffect(() => {
    if (!lineage) return
    let gone = false
    void api
      .functions({ lineage, limit: 500 })
      .then((page) => {
        if (!gone) setState({ key: lineage, uploads: page.items })
      })
      .catch(() => {
        if (!gone) setState({ key: lineage, uploads: [] })
      })
    return () => {
      gone = true
    }
  }, [lineage])
  return { uploads: state.key === lineage ? state.uploads : [], loading: !!lineage && state.key !== lineage }
}
