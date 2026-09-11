import { useMemo } from 'react'
import { openWizard } from '../../sheets'
import { useStore } from '../../state/store'
import { money } from '../../state/model'
import type { Offer } from '../../api/client'
import { Chip, Pick } from '../../ui/primitives'

const ondemand = (o: Offer): number => o.on_demand_price ?? o.price ?? 0
const spot = (o: Offer): number | null => o.spot_price ?? null
const accelOf = (o: Offer): string => o.accelerator ?? 'cpu'
const availOf = (o: Offer): number => o.available ?? 0
const fetchedAgo = (o: Offer): number => Math.max(0, Math.round((Date.now() - Date.parse(o.fetched_at)) / 60000))

export function Stage() {
  const offers = useStore((s) => s.offers)
  const f = useStore((s) => s.market)
  const setUi = useStore((s) => s.setUi)

  const accels = useMemo(() => ['all', ...new Set(offers.map(accelOf))], [offers])

  const per = (o: Offer): number => (f.market === 'spot' ? (spot(o) ?? 0) : (spot(o) ?? ondemand(o))) / (o.accelerator_count || 1)

  const rows = offers
    .filter((o) => (f.accel === 'all' || accelOf(o) === f.accel) && (f.market === 'all' || spot(o) != null))
    .sort((a, b) =>
      f.sort === 'price' ? per(a) - per(b) : f.sort === 'vram' ? (b.vram ?? 0) - (a.vram ?? 0) : availOf(b) - availOf(a),
    )
  const cheapest = rows.length ? per(rows[0]!) : 1

  return (
    <section className="card">
      <div className="combhead">
        <b>Market</b>
        <span className="sub">{offers.length} offers</span>
      </div>
      <div className="row wrap" style={{ gap: 8, marginBottom: 10 }}>
        <div className="chips">
          {accels.map((a) => (
            <Chip key={a} pressed={f.accel === a} onClick={() => setUi({ market: { ...f, accel: a } })}>
              {a === 'all' ? 'any GPU' : a.toUpperCase()}
            </Chip>
          ))}
        </div>
        <div style={{ marginLeft: 'auto' }}>
          <Pick
            value={f.market}
            options={[
              ['all', 'either price'],
              ['spot', 'spot only'],
            ]}
            onChange={(v) => setUi({ market: { ...f, market: v } })}
          />
        </div>
        <Pick
          value={f.sort}
          options={[
            ['price', 'cheapest'],
            ['vram', 'most VRAM'],
            ['avail', 'most available'],
          ]}
          onChange={(v) => setUi({ market: { ...f, sort: v } })}
        />
      </div>
      <div className="scroll">
        <table>
          <thead>
            <tr>
              <th>Provider</th>
              <th>Instance</th>
              <th>Shape</th>
              <th>Region</th>
              <th className="right">Spot</th>
              <th className="right">On demand</th>
              <th>Per GPU·h</th>
              <th>Available</th>
              <th>Fetched</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {rows.map((o) => {
              const p = per(o)
              const rel = p / (cheapest || 1)
              const available = availOf(o)
              return (
                <tr key={o.id}>
                  <td>
                    <b style={{ fontWeight: 600 }}>{o.kind}</b>
                  </td>
                  <td className="mono">{o.instance_type}</td>
                  <td className="sub">
                    {o.accelerator_count}× {accelOf(o).toUpperCase()} · {o.cpus} vCPU · {o.memory_gb} GB
                  </td>
                  <td className="mono faint">{o.region ?? '—'}</td>
                  <td className="right mono">{spot(o) != null ? money(spot(o)!) : <span className="faint">—</span>}</td>
                  <td className="right mono">{money(ondemand(o))}</td>
                  <td>
                    <div className="row" style={{ gap: 7 }}>
                      <span className="mono" style={{ width: 48 }}>
                        {money(p)}
                      </span>
                      <span className="track" style={{ width: 64 }}>
                        <i style={{ width: `${Math.min(100, 100 / (rel || 1))}%` }} />
                      </span>
                    </div>
                  </td>
                  <td className="mono faint">{available === 0 ? <span style={{ color: 'var(--bad)' }}>none</span> : available}</td>
                  <td className="faint">{fetchedAgo(o)}m ago</td>
                  <td className="right">
                    <button className="btn sm" disabled={available === 0} onClick={() => openWizard(o)}>
                      Use
                    </button>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </section>
  )
}
