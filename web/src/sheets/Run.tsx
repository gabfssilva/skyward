import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, type Call, type FunctionRef, type TaskCreate } from '../api/client'
import { nodesOf, useLibrary, useStore } from '../state/store'
import { useLineage, versionsOf, whereOf } from '../state/functions'
import { readyOf } from '../state/model'
import { Icon } from '../ui/icons'
import { Pick } from '../ui/primitives'
import { Scrim, CloseBtn } from './Scrim'

type Dispatch = TaskCreate['dispatch']

const ANY = 'any'
const LATEST = 'latest'

const record = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null && !Array.isArray(value)

/**
 * The arguments, read from the one box they were typed in.
 *
 * A list is positional and an object is by name, which is the same distinction the
 * wire draws between ``args`` and ``kwargs`` — so the shape of what was typed says
 * which it is, and there is nothing to choose between two boxes that would each be
 * empty half the time.
 */
const called = (text: string): Call => {
  const written: unknown = JSON.parse(text.trim() || '[]')
  if (Array.isArray(written)) return { args: written }
  if (record(written)) return { kwargs: written }
  throw new Error('arguments are a list, taken in order, or an object, taken by name')
}

const labelOf = (fn: FunctionRef): string => `${fn.qualname ?? fn.name ?? fn.sha256.slice(0, 8)} · ${whereOf(fn)}`

/**
 * One task: a function, a version of it, a compute, and one machine or all of them.
 *
 * The version is the latest unless one is asked for, and "latest" is decided when
 * the task is submitted rather than when the sheet opened. What the task records is
 * the exact upload it ran, so asking later what ran is never answered with "latest".
 */
export function Run({ lineage, version, computeId, node }: { lineage?: string; version?: number; computeId?: string; node?: number }) {
  const navigate = useNavigate()
  const close = useStore((s) => s.closeSheet)
  const reloadCompute = useStore((s) => s.reloadCompute)
  const computes = useStore((s) => s.computes)
  const state = useStore((s) => s)
  const library = useLibrary()
  const [picked, setPicked] = useState<string>(lineage ?? '')
  const [wanted, setWanted] = useState<string>(version === undefined ? LATEST : String(version))
  const [compute, setCompute] = useState(computeId ?? computes[0]?.id ?? '')
  const [dispatch, setDispatch] = useState<Dispatch>('one')
  const [rank, setRank] = useState<string>(node === undefined ? ANY : String(node))
  const [args, setArgs] = useState('[]')
  const [refusal, setRefusal] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const held = library?.items ?? []
  const key = picked || held[0]?.lineage || ''
  const { uploads } = useLineage(key || null)
  const versions = versionsOf(uploads)
  const chosen = wanted === LATEST ? versions[0] : versions.find((v) => String(v.version) === wanted)
  const sha256 = chosen?.newest.sha256 ?? ''

  /* A select shows its first option for a value none of them carry, which would submit one thing while naming another. */
  const unlisted = key !== '' && !held.some((fn) => fn.lineage === key) ? uploads[0] : undefined
  const ready = compute ? readyOf(nodesOf(state, compute)) : []
  const at = ready.some((n) => String(n.rank) === rank) ? rank : ANY

  const submit = async () => {
    setBusy(true)
    setRefusal(null)
    try {
      const task = await api.submitTask({
        compute,
        function: sha256,
        dispatch,
        rank: dispatch === 'one' && at !== ANY ? Number(at) : null,
        call: called(args),
      })
      close()
      void reloadCompute(compute)
      navigate(`/tasks/${task.id}`)
    } catch (error) {
      setRefusal(error instanceof Error ? error.message : String(error))
      setBusy(false)
    }
  }

  return (
    <Scrim label="Run">
      <div className="sheet" style={{ width: 'min(560px,100%)' }}>
        <div className="sheet-head">
          <b>Run a function</b>
          <CloseBtn style={{ marginLeft: 'auto' }} />
        </div>
        <div className="sheet-body" style={{ display: 'grid', gap: 10 }}>
          <div className="fnpick">
            <div className="field">
              <label htmlFor="run-fn">Function</label>
              <select
                id="run-fn"
                className="search"
                value={key}
                onChange={(e) => {
                  setPicked(e.target.value)
                  setWanted(LATEST)
                }}
              >
                {held.length ? null : <option value="">nothing is registered yet</option>}
                {unlisted ? <option value={key}>{labelOf(unlisted)}</option> : null}
                {held.map((fn) => (
                  <option key={fn.lineage} value={fn.lineage}>
                    {labelOf(fn)}
                  </option>
                ))}
              </select>
            </div>
            <div className="field">
              <label htmlFor="run-version">Version</label>
              <select id="run-version" className="search" value={wanted} onChange={(e) => setWanted(e.target.value)}>
                <option value={LATEST}>latest{versions[0] ? ` · v${versions[0].version}` : ''}</option>
                {versions.map((v) => (
                  <option key={v.version} value={String(v.version)}>
                    v{v.version}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <div className="field">
            <label htmlFor="run-compute">Compute</label>
            <select id="run-compute" className="search" value={compute} onChange={(e) => setCompute(e.target.value)}>
              {computes.length ? null : <option value="">no compute is running</option>}
              {computes.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name ?? c.id} · {readyOf(nodesOf(state, c.id)).length} ready
                </option>
              ))}
            </select>
          </div>
          <div className="row wrap" style={{ gap: 8 }}>
            <Pick<Dispatch>
              value={dispatch}
              options={[
                ['one', 'one node'],
                ['all', 'every node'],
              ]}
              onChange={setDispatch}
            />
            {dispatch === 'one' ? (
              <select className="search" aria-label="node" style={{ marginLeft: 'auto', minWidth: 140 }} value={at} onChange={(e) => setRank(e.target.value)}>
                <option value={ANY}>any node</option>
                {ready.map((n) => (
                  <option key={n.rank} value={n.rank}>
                    node {n.rank}
                  </option>
                ))}
              </select>
            ) : (
              <span className="sub" style={{ marginLeft: 'auto' }}>
                one execution per ready node, frozen now
              </span>
            )}
          </div>
          <div className="field">
            <label htmlFor="run-args">Arguments</label>
            <textarea id="run-args" className="code" spellCheck={false} rows={4} value={args} onChange={(e) => setArgs(e.target.value)} />
            <span className="sub">JSON. A list is taken in order, an object by name — anything a dataframe has to be built on the machine.</span>
          </div>
          {refusal ? (
            <div className="strip bad" style={{ alignItems: 'flex-start' }}>
              <Icon name="alert" />
              <span className="mono">{refusal}</span>
            </div>
          ) : null}
        </div>
        <div className="sheet-foot">
          <button className="btn primary" disabled={busy || !compute || !sha256} onClick={() => void submit()}>
            <Icon name="run" />
            Run
          </button>
        </div>
      </div>
    </Scrim>
  )
}
