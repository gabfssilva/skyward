import { useState } from 'react'
import { useStore } from '../state/store'
import { useLineage, versionsOf, whereOf } from '../state/functions'
import { ago, ms } from '../state/model'
import { Icon } from '../ui/icons'
import { Scrim, CloseBtn } from './Scrim'

/**
 * One function: its versions, and the code of the chosen one.
 *
 * A pickle is compiled code and carries no text, so what is shown comes from where
 * the function was defined. One written in the console is its own text, and can be
 * edited into a new version. One the SDK uploaded is what the SDK read off its file
 * — the function and what it uses from its module — which is for reading: the file
 * is the place to change it.
 */
export function Function({ lineage }: { lineage: string }) {
  const openSheet = useStore((s) => s.openSheet)
  const { uploads, loading } = useLineage(lineage)
  const versions = versionsOf(uploads)
  const [picked, setPicked] = useState<number | null>(null)
  const chosen = versions.find((v) => v.version === picked) ?? versions[0]
  const fn = chosen?.newest
  const latest = versions[0] === chosen
  const text = fn?.source ?? fn?.excerpt

  return (
    <Scrim label="Function" dismissible>
      <div className="sheet" style={{ width: 'min(820px,100%)' }}>
        <div className="sheet-head">
          <b>{fn?.qualname ?? fn?.name ?? 'Function'}</b>
          {fn ? <span className="tag">v{fn.version}</span> : null}
          {fn ? <span className="sub">{whereOf(fn)}</span> : null}
          <CloseBtn style={{ marginLeft: 'auto' }} />
        </div>
        <div className="sheet-body" style={{ display: 'grid', gap: 12 }}>
          {!fn ? (
            <span className="sub">{loading ? 'Reading its versions…' : 'No such function.'}</span>
          ) : (
            <>
              {fn.origin && !fn.source ? <div className="mono faint">{fn.origin}</div> : null}
              <div className="row" style={{ gap: 8 }}>
                <select className="search" aria-label="version" value={chosen.version} onChange={(e) => setPicked(Number(e.target.value))}>
                  {versions.map((v, i) => (
                    <option key={v.version} value={v.version}>
                      v{v.version}
                      {i === 0 ? ' · latest' : ''} · {v.uploads.length} upload{v.uploads.length === 1 ? '' : 's'} · {ago(ms(v.newest.created_at))}
                    </option>
                  ))}
                </select>
                <div className="row" style={{ gap: 6, marginLeft: 'auto' }}>
                  {fn.source ? (
                    <button className="btn sm" onClick={() => openSheet({ kind: 'write', from: fn })}>
                      Edit
                    </button>
                  ) : null}
                  <button className="btn sm primary" onClick={() => openSheet({ kind: 'run', lineage, version: latest ? undefined : chosen.version })}>
                    <Icon name="run" />
                    Run {latest ? 'latest' : `v${chosen.version}`}
                  </button>
                </div>
              </div>
              {text ? (
                <pre className="codebox" style={{ maxHeight: 'min(560px, 60svh)', whiteSpace: 'pre' }}>
                  {text}
                </pre>
              ) : (
                <div className="strip" style={{ alignItems: 'flex-start', flexDirection: 'column', gap: 3 }}>
                  <b style={{ fontWeight: 600 }}>No code to show.</b>
                  <span className="sub">Its text was not sent with it: it was read from stdin, built by an exec, typed at a REPL older than Python 3.13, taken from a file gone by the time it was sent, or uploaded straight to the API.</span>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </Scrim>
  )
}
