import type { FunctionRef } from '../../api/client'
import { useLibrary, useStore } from '../../state/store'
import { whereOf } from '../../state/functions'
import { ago, ms } from '../../state/model'
import { Icon } from '../../ui/icons'
import { TableScroll } from '../../ui/primitives'
import { openFunction, openRun, openWrite } from '../../sheets'

const NONE: never[] = []

function Row({ fn }: { fn: FunctionRef }) {
  return (
    <tr style={{ cursor: 'pointer' }} onClick={() => openFunction(fn.lineage)}>
      <td>
        <b>{fn.qualname ?? fn.name ?? <span className="mono">{fn.sha256.slice(0, 8)}</span>}</b>
      </td>
      <td className="mono">v{fn.version}</td>
      <td className="mono faint">{whereOf(fn)}</td>
      <td className="mono faint">{ago(ms(fn.created_at))}</td>
      <td className="right">
        <button
          className="btn sm"
          onClick={(e) => {
            e.stopPropagation()
            openRun({ lineage: fn.lineage })
          }}
        >
          <Icon name="run" />
          Run
        </button>
      </td>
    </tr>
  )
}

/**
 * The Functions view: one row per function, at its latest version.
 *
 * A function is many uploads — every edit, and every run that captured something
 * different — so a row is the function and not the upload. Opening one shows its
 * versions and what each is made of; running one runs its latest unless another is
 * asked for.
 */
export function Stage() {
  const library = useLibrary()
  const pageLibrary = useStore((s) => s.pageLibrary)
  const rows = library?.items ?? NONE
  const total = library?.total ?? null
  return (
    <section className="card">
      <div className="combhead">
        <b>Functions</b>
        <span className="sub">{total === null ? rows.length : `${rows.length} of ${total}`}</span>
        <button className="btn primary" style={{ marginLeft: 'auto' }} onClick={() => openWrite()}>
          <Icon name="plus" />
          New function
        </button>
      </div>
      <TableScroll label="Functions">
        <table className="cards fnlist">
          <thead>
            <tr>
              <th>Function</th>
              <th>Version</th>
              <th>Written in</th>
              <th>Last sent</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {rows.length ? (
              rows.map((fn) => <Row key={fn.sha256} fn={fn} />)
            ) : (
              <tr>
                <td colSpan={5} className="sub" style={{ padding: '22px 8px', textAlign: 'center' }}>
                  {(library?.loading ?? true) ? 'Reading functions…' : 'Nothing is registered yet. Write one, or dispatch from the SDK.'}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </TableScroll>
      {library?.cursor ? (
        <button className="btn sm" style={{ marginTop: 10 }} disabled={library.loading} onClick={() => void pageLibrary()}>
          Show older functions
        </button>
      ) : null}
    </section>
  )
}
