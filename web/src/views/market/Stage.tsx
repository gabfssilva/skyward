import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, type Offer, type Provider } from '../../api/client'
import { useOffers, useStore } from '../../state/store'
import type { MarketFilters } from '../../state/store'
import { ago, money, ms } from '../../state/model'
import { ACCELS } from '../../sheets/catalog'
import { openAddProvider, openConfirm, openWizard } from '../../sheets'
import { Actions, PageHead, Tabs } from '../../ui/head'
import { Chip, Pick, Pill, TableScroll } from '../../ui/primitives'

type Tab = 'offers' | 'accounts'

const ondemand = (o: Offer): number => o.on_demand_price ?? o.price ?? 0
const spot = (o: Offer): number | null => o.spot_price ?? null
const availOf = (o: Offer): number => o.available ?? 0

/**
 * What a chip can ask the daemon for.
 *
 * The rows are one slice of an order the daemon cut, so they are no vocabulary of their own:
 * chips read off them would wobble as the market pages. The app's own accelerators are the
 * vocabulary, and an accelerator the loaded rows carry that it does not name is added to it —
 * an accelerator a provider sells is never unreachable.
 */
const vocabulary = (rows: readonly Offer[]): string[] => {
  const known = Object.keys(ACCELS)
  const extra = rows.map((o) => o.accelerator).filter((a): a is string => !!a && !known.includes(a))
  return ['all', ...known, ...[...new Set(extra)].sort()]
}

/** A column that orders the listing, marked when it is the one the daemon ordered by. */
function Sortable({ label, by, right }: { label: string; by: MarketFilters['sort']; right?: boolean }) {
  const f = useStore((s) => s.market)
  const setUi = useStore((s) => s.setUi)
  return (
    <th className={right ? 'right' : undefined}>
      <button style={{ color: f.sort === by ? 'var(--ink)' : 'inherit', fontWeight: 600 }} onClick={() => setUi({ market: { ...f, sort: by } })}>
        {label}
      </button>
    </th>
  )
}

/** What can be bought right now, cheapest per accelerator-hour first. */
function Offers() {
  const rows = useStore((s) => s.offers)
  const catalog = useOffers()
  const f = useStore((s) => s.market)
  const setUi = useStore((s) => s.setUi)
  const moreOffers = useStore((s) => s.moreOffers)
  const accels = useMemo(() => vocabulary(rows), [rows])
  const per = (o: Offer): number => (f.market === 'spot' ? (spot(o) ?? 0) : (spot(o) ?? ondemand(o))) / (o.accelerator_count || 1)
  const cheapest = rows.length ? per(rows[0]!) : 1

  return (
    <section className="card">
      <div className="row wrap" style={{ gap: 8, marginBottom: 10 }}>
        <div className="chips">
          {accels.map((a) => (
            <Chip key={a} pressed={f.accel === a} onClick={() => setUi({ market: { ...f, accel: a } })}>
              {a === 'all' ? 'Any accelerator' : a.toUpperCase()}
            </Chip>
          ))}
        </div>
        <div className="spread">
          <Pick
            value={f.market}
            options={[
              ['all', 'Either price'],
              ['spot', 'Spot only'],
            ]}
            onChange={(v) => setUi({ market: { ...f, market: v } })}
          />
        </div>
      </div>
      <TableScroll label="Offers">
        <table>
          <thead>
            <tr>
              <Sortable label="Machine" by="vram" />
              <th>Where</th>
              <th className="right">Spot</th>
              <th className="right">On demand</th>
              <Sortable label="Per accelerator·h" by="price" />
              <Sortable label="Available" by="available" right />
              <th />
            </tr>
          </thead>
          <tbody>
            {rows.map((o) => {
              const p = per(o)
              const available = availOf(o)
              return (
                <tr key={o.id} data-tip={`fetched ${ago(ms(o.fetched_at))}`}>
                  <td>
                    <span className="mono">{o.instance_type}</span>
                    <span className="sub">
                      {o.accelerator ? `${o.accelerator_count}× ${o.accelerator.toUpperCase()}, ` : ''}
                      {o.cpus} vCPU, {o.memory_gb} GB
                    </span>
                  </td>
                  <td className="nowrap">
                    {o.kind}
                    <span className="sub">{o.region ?? 'any region'}</span>
                  </td>
                  <td className="right nowrap">{spot(o) != null ? money(spot(o)!) : <span className="faint">—</span>}</td>
                  <td className="right nowrap">{money(ondemand(o))}</td>
                  <td className="nowrap">
                    <b style={{ fontWeight: 600 }}>{money(p)}</b>
                    <span className="track" style={{ marginLeft: 10 }}>
                      <i style={{ width: `${Math.min(100, (cheapest / (p || 1)) * 100)}%` }} />
                    </span>
                  </td>
                  <td className="right">{available === 0 ? <span style={{ color: 'var(--bad)' }}>none</span> : available}</td>
                  <td className="right">
                    <button className="btn sm ghost" disabled={available === 0} onClick={() => openWizard(o)}>
                      Use
                    </button>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </TableScroll>
      {catalog && catalog.total !== null && rows.length < catalog.total ? (
        <button className="btn sm" style={{ marginTop: 10 }} disabled={catalog.loading} onClick={() => void moreOffers()}>
          Show more
        </button>
      ) : null}
    </section>
  )
}

/** The accounts the offers come from: whether the daemon can still use each one, and what it last answered with. */
function Accounts() {
  const providers = useStore((s) => s.providers)
  const reloadProviders = useStore((s) => s.reloadProviders)
  const reloadOffers = useStore((s) => s.reloadOffers)
  const [busy, setBusy] = useState<string | null>(null)

  const fetchOffers = async (p: Provider) => {
    setBusy(p.id)
    try {
      await api.offers({ provider: p.id, refresh: true })
      await Promise.all([reloadOffers(), reloadProviders()])
    } finally {
      setBusy(null)
    }
  }

  const check = async (p: Provider) => {
    setBusy(p.id)
    try {
      await api.provider(p.id)
      await reloadProviders()
    } finally {
      setBusy(null)
    }
  }

  const remove = (p: Provider) =>
    openConfirm({
      title: 'Remove this account?',
      body: `${p.kind} ${p.name} will stop being priced, and computes it holds will keep running.`,
      confirm: 'Remove',
      danger: true,
      onConfirm: () => {
        void api.deleteProvider(p.id).then(reloadProviders)
      },
    })

  return (
    <section className="card">
      <TableScroll label="Accounts">
        <table>
          <thead>
            <tr>
              <th>Account</th>
              <th>State</th>
              <th className="right">Offers</th>
              <th className="right">Refreshed</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {providers.map((p) => (
              <tr key={p.id}>
                <td>
                  <b style={{ fontWeight: 600 }}>{p.kind}</b>
                  {p.name && p.name !== 'default' ? <span className="sub" style={{ display: 'inline' }}> {p.name}</span> : null}
                </td>
                <td style={{ maxWidth: 340 }}>
                  <Pill state={p.last_error ? 'failed' : 'ready'} />
                  {p.last_error ? <span className="err-line">{p.last_error.message}</span> : null}
                </td>
                <td className="right">{p.offers_count ? p.offers_count.toLocaleString('en-US') : <span className="faint">—</span>}</td>
                <td className="right sub nowrap">{p.offers_fetched_at ? ago(ms(p.offers_fetched_at)) : '—'}</td>
                <td>
                  <div className="row" style={{ justifyContent: 'flex-end' }}>
                    <Actions
                      rest={[
                        { label: 'Check', disabled: busy === p.id, onClick: () => void check(p) },
                        { label: 'Fetch offers', disabled: busy === p.id, onClick: () => void fetchOffers(p) },
                        { label: 'Remove account', icon: 'trash', danger: true, onClick: () => remove(p) },
                      ]}
                    />
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </TableScroll>
    </section>
  )
}

/** The market: what can be bought, and the accounts it can be bought through. */
export function Stage({ accounts }: { accounts?: boolean }) {
  const navigate = useNavigate()
  const offers = useStore((s) => s.offers)
  const catalog = useOffers()
  const providers = useStore((s) => s.providers)
  const tab: Tab = accounts ? 'accounts' : 'offers'

  return (
    <>
      <PageHead
        primary={tab === 'accounts' ? { label: 'Add account', icon: 'plus', onClick: () => openAddProvider() } : undefined}
      >
        <div className="tabrow">
          <Tabs<Tab>
            value={tab}
            options={[
              ['offers', 'Offers', catalog?.total ?? offers.length],
              ['accounts', 'Accounts', providers.length],
            ]}
            onChange={(next) => navigate(next === 'accounts' ? '/market/accounts' : '/market')}
          />
        </div>
      </PageHead>
      {tab === 'accounts' ? <Accounts /> : <Offers />}
    </>
  )
}
