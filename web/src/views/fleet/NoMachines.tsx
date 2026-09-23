import { useNavigate } from 'react-router-dom'
import type { Compute } from '../../api/client'
import { useStore } from '../../state/store'
import { boundOf, boundsOf, callsOf, failedOf, machineOf, money } from '../../state/model'
import { Pill, TableScroll } from '../../ui/primitives'
import { hardwareOf, outcomeOf } from './History'

/**
 * The computes that hold no machine: a row each, in the columns the ended ones are read in.
 *
 * A tile is a hive with a caption, and a compute of no nodes has no hive — as a tile it was a box kept empty.
 * A row opens it, as a tile does.
 */
export function NoMachines({ computes }: { computes: readonly Compute[] }) {
  const navigate = useNavigate()
  const costs = useStore((s) => s.costs)

  return (
    <section className="card nomach">
      <div className="chead" style={{ marginBottom: 4 }}>
        <span className="h">No machines</span>
        <span className="sub">{computes.length}</span>
      </div>
      <TableScroll label="No machines">
        <table>
          <thead>
            <tr>
              <th>Compute</th>
              <th>State</th>
              <th className="wide">Provider</th>
              <th className="wide">Specs</th>
              <th className="wide right">Nodes</th>
              <th className="wide right">Calls</th>
              <th className="right">Spent</th>
            </tr>
          </thead>
          <tbody>
            {computes.map((c) => {
              const bound = boundOf(c)
              const under = hardwareOf(c)
              const spent = costs[c.id] ?? 0
              return (
                <tr key={c.id} data-open="" onClick={() => navigate(`/computes/${c.id}`)}>
                  <td>
                    <b style={{ fontWeight: 600 }}>{c.name ?? c.id}</b>
                    <span className="sub mono">{c.id}</span>
                  </td>
                  <td className="nowrap">
                    <Pill state={c.status.state} />
                  </td>
                  <td className="wide nowrap">
                    {bound?.kind ?? '—'}
                    <span className="sub">{bound?.region ?? 'any region'}</span>
                  </td>
                  <td className="wide nowrap">
                    {machineOf(c)}
                    <span className={under.code ? 'sub mono' : 'sub'}>{under.text}</span>
                  </td>
                  <td className="wide right nowrap">
                    0<span className="sub">{boundsOf(c)}</span>
                  </td>
                  <td className="wide right nowrap">
                    {callsOf(c) || '—'}
                    <span className="sub" style={{ color: failedOf(c) ? 'var(--bad)' : undefined }}>{outcomeOf(c)}</span>
                  </td>
                  <td className="right nowrap">{spent ? money(spent, spent < 10 ? 2 : 0) : '—'}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </TableScroll>
    </section>
  )
}
