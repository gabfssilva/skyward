import { useLibrary } from '../../state/store'
import { Icon } from '../../ui/icons'
import { openWrite } from '../../sheets'

/** What the library is made of: how many functions, and how many of them can be read back as text. */
export function Inspector() {
  const library = useLibrary()
  const rows = library?.items ?? []
  const written = rows.filter((fn) => fn.source).length
  return (
    <>
      <section className="card tight">
        <div className="cap">The library</div>
        <div style={{ marginTop: 6 }}>
          <div className="kv">
            <span className="faint">functions</span>
            <span className="mono">{library?.total ?? rows.length}</span>
          </div>
          <div className="kv">
            <span className="faint">written here</span>
            <span className="mono">{written}</span>
          </div>
          <div className="kv">
            <span className="faint">sent by the SDK</span>
            <span className="mono">{rows.length - written}</span>
          </div>
        </div>
        <button className="btn sm" style={{ marginTop: 10 }} onClick={() => openWrite()}>
          <Icon name="plus" />
          New function
        </button>
      </section>
      <section className="card tight">
        <div className="cap">Versions</div>
        <div className="sub" style={{ marginTop: 6 }}>
          A function is the same name in the same file. Its version moves when its code does — not when it moves down the file, and not when a run
          captures a different value — and running it runs the latest unless you pick another.
        </div>
      </section>
    </>
  )
}
