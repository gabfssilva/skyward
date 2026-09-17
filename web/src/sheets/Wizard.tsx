import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import type { CSSProperties, ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import type { Accelerator, ComputeCreate, ComputeSpec, Offer, Provider, Worker } from '../api/client'
import { useStore } from '../state/store'
import { money } from '../state/model'
import { Icon } from '../ui/icons'
import { Chip, Pick, Tick } from '../ui/primitives'
import { Scrim, CloseBtn } from './Scrim'
import { COLLECTIVE, PLUGINS } from './catalog'

const STEPS = 3

type Allocation = ComputeSpec['allocation']
type Executor = NonNullable<Worker['executor']>
type Line = 'datacenter' | 'pro' | 'consumer' | 'other'

type Draft = {
  step: number
  /** Provider ids; none chosen is every account. */
  providers: readonly string[]
  accel: string
  /** Accelerators per node; zero asks for none. */
  count: number
  cpus: number | null
  memory: number | null
  initial: number
  min: number | null
  max: number | null
  allocation: Allocation
  base: string
  python: string
  pip: readonly string[]
  plugins: readonly string[]
  executor: Executor
  concurrency: number
  reuse: boolean
  name: string
  ttl: number
}

type Shelf =
  | { state: 'loading' }
  | { state: 'failed'; message: string }
  | { state: 'ready'; offers: readonly Offer[]; catalog: ReadonlyMap<string, Accelerator> }

type Buy = { offer: Offer; market: 'spot' | 'on_demand'; price: number }

type Model = { name: string; label: string; vram: number; make: string; line: Line; from: number }

type Market = {
  models: readonly Model[]
  /** The accelerator asked for, or the nearest in VRAM when the chosen accounts stopped selling it. */
  accel: string
  buys: readonly Buy[]
  choice: Buy | null
  /** How many accelerators each account sells at the chosen allocation. */
  sold: ReadonlyMap<string, number>
}

const ALLOCS: readonly (readonly [Allocation, string])[] = [
  ['spot_if_available', 'spot if available'],
  ['spot', 'spot only'],
  ['on_demand', 'on demand'],
  ['cheapest', 'cheapest'],
]

const LINES: readonly (readonly [Line, string])[] = [
  ['datacenter', 'Data center'],
  ['pro', 'Pro · workstation'],
  ['consumer', 'Consumer'],
  ['other', 'Unclassified'],
]

/** The adapters that read `Image.base`: they run a container, the rest boot a machine image of their own. */
const CONTAINERS: ReadonlySet<string> = new Set(['runpod', 'vastai', 'novita', 'salad'])

const DATACENTER: ReadonlySet<string> = new Set([
  'a10', 'a100', 'a100x', 'a10g', 'a16', 'a2', 'a30', 'a40', 'a800', 'b100', 'b200', 'b300', 'gb200', 'gb300', 'gh200',
  'h100', 'h100-nvl', 'h200', 'h200-nvl', 'k80', 'l4', 'l40', 'l40s', 'p100', 'p4', 'p40', 't4', 't4g', 'v100',
])

/** The names no rule reads well: a word order the catalog does not keep, and salad's bundles of several cards. */
const LABELS: Readonly<Record<string, string>> = {
  'rtx-pro-server-6000': 'RTX PRO 6000 Server',
  gtx107010801080ti: 'GTX 1070 / 1080 / 1080 Ti',
  stablediffusioncompatible: 'Any SD-capable GPU',
}

const WORDS: Readonly<Record<string, string>> = {
  ti: 'Ti', super: 'Super', ada: 'Ada', laptop: 'Laptop', maxq: 'Max-Q', wk: 'Workstation', sff: 'SFF', xt: 'XT', xtx: 'XTX',
}

const DUO: CSSProperties = { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }
const TRIO: CSSProperties = { display: 'grid', gridTemplateColumns: 'repeat(3,minmax(0,120px))', gap: 10, marginTop: 8 }
const STACK: CSSProperties = { display: 'flex', flexDirection: 'column', minHeight: '100%' }

const newDraft = (offer: Offer | undefined): Draft => ({
  step: 1,
  providers: offer ? [offer.provider_id] : [],
  accel: offer?.accelerator ?? 'h100',
  count: offer ? (offer.accelerator ? offer.accelerator_count : 0) : 1,
  cpus: null,
  memory: null,
  initial: 1,
  min: null,
  max: null,
  allocation: 'spot_if_available',
  base: '',
  python: '3.12',
  pip: [],
  plugins: [],
  executor: 'thread',
  concurrency: 1,
  reuse: true,
  name: '',
  ttl: 600,
})

/** The offer the wizard opens on, handed over by the market view through `openWizard`. */
let seed: Offer | undefined

export const seedWizard = (offer: Offer | undefined): void => {
  seed = offer
}

export function Wizard() {
  const navigate = useNavigate()
  const closeSheet = useStore((s) => s.closeSheet)
  const load = useStore((s) => s.load)
  const accounts = useStore((s) => s.providers)
  const [w, setW] = useState<Draft>(() => newDraft(seed))
  const [shelf, setShelf] = useState<Shelf>({ state: 'loading' })
  const [busy, setBusy] = useState(false)
  /** Why the last create was refused; any edit clears it, since the draft it was about is gone. */
  const [failure, setFailure] = useState<string | null>(null)
  const patch = (p: Partial<Draft>) => {
    setFailure(null)
    setW((prev) => ({ ...prev, ...p }))
  }

  useEffect(() => {
    let live = true
    Promise.all([api.offers(), api.accelerators()])
      .then(([page, catalog]) => {
        if (live) setShelf({ state: 'ready', offers: page.items, catalog: new Map(catalog.map((a) => [a.name, a])) })
      })
      .catch((error: unknown) => {
        if (live) setShelf({ state: 'failed', message: error instanceof Error ? error.message : String(error) })
      })
    return () => {
      live = false
    }
  }, [])

  const m = useMemo(() => (shelf.state === 'ready' ? market(shelf.offers, shelf.catalog, w) : market([], new Map(), w)), [shelf, w])
  const tone = (providerId: string): string => `var(--c${(Math.max(0, accounts.findIndex((p) => p.id === providerId)) % 5) + 1})`

  const create = async () => {
    setBusy(true)
    try {
      const chosen = w.providers.length ? accounts.filter((p) => w.providers.includes(p.id)) : accounts
      const payload: ComputeCreate = {
        name: w.name || null,
        spec: {
          allocation: w.allocation,
          selection: 'cheapest',
          delete_on_exit: false,
          desired: 'running',
          ttl: w.ttl,
          nodes: { initial: w.initial, min: w.min, max: w.max },
          specs: chosen.map((p) => ({
            provider: { kind: p.kind, name: p.name },
            accelerator: w.count ? m.accel : null,
            accelerator_count: w.count || 1,
            cpus: w.cpus,
            memory_gb: w.memory,
          })),
          plugins: w.plugins.map((kind) => ({ kind, params: {} })),
          volumes: [],
          image: {
            base: w.base || null,
            python: w.python || null,
            pip: [...w.pip],
            apt: [],
            excludes: [],
            includes: [],
            pip_indexes: [],
            bootstrap_timeout: 900,
            skyward: 'auto',
            warm: false,
          },
          worker: { executor: w.executor, concurrency: w.concurrency, buffer: 0, reuse: w.executor === 'process' ? w.reuse : true },
        },
      }
      const compute = await api.createCompute(payload)
      closeSheet()
      await load()
      navigate(`/computes/${compute.id}`)
    } catch (error) {
      setFailure(error instanceof Error ? error.message : String(error))
    } finally {
      setBusy(false)
    }
  }

  const next = () => {
    if (w.step === STEPS) void create()
    else patch({ step: w.step + 1 })
  }

  return (
    <Scrim>
      <div className="sheet wizard" role="dialog" aria-label="New compute">
        <div className="sheet-head">
          <b>New compute</b>
          <div className="steps">
            {Array.from({ length: STEPS }, (_, i) => (
              <i key={i} className={i < w.step ? 'on' : ''} />
            ))}
          </div>
          <CloseBtn />
        </div>
        <div className="sheet-body">
          {w.step === 1 ? <Need w={w} patch={patch} m={m} shelf={shelf} accounts={accounts} tone={tone} /> : null}
          {w.step === 2 ? <Runs w={w} patch={patch} m={m} /> : null}
          {w.step === 3 ? <Review w={w} m={m} shelf={shelf} tone={tone} /> : null}
        </div>
        <div className="sheet-foot">
          {w.step > 1 ? (
            <button className="btn" onClick={() => patch({ step: w.step - 1 })}>
              <Icon name="back" />
              Back
            </button>
          ) : null}
          {failure ? (
            <span className="sub trunc" style={{ color: 'var(--bad)', minWidth: 0 }} data-tip={failure}>
              {failure}
            </span>
          ) : (
            <span className="sub">{m.choice ? `${money(m.choice.price * w.initial)} per hour` : shelf.state === 'ready' ? 'No offer fits' : ''}</span>
          )}
          <button className="btn primary" style={{ marginLeft: 'auto' }} disabled={busy || !m.choice} onClick={next}>
            {w.step === STEPS ? (
              <>
                <Icon name="check" />
                Create compute
              </>
            ) : (
              'Continue'
            )}
          </button>
        </div>
      </div>
    </Scrim>
  )
}

/**
 * The market the daemon would buy from, read the way `market._candidates` reads it: over one spec per chosen
 * account, the accelerator count is matched exactly and vCPUs and RAM as floors. Asking for no accelerator filters on
 * nothing but those floors.
 */
function market(offers: readonly Offer[], catalog: ReadonlyMap<string, Accelerator>, w: Draft): Market {
  const inAccounts = (o: Offer) => !w.providers.length || w.providers.includes(o.provider_id)
  const buyable = offers.filter((o) => buysOf(o, w.allocation).length)

  const grouped = new Map<string, Offer[]>()
  const sold = new Map<string, Set<string>>()
  for (const o of buyable) {
    if (!o.accelerator) continue
    sold.set(o.provider_id, (sold.get(o.provider_id) ?? new Set()).add(o.accelerator))
    if (inAccounts(o)) grouped.set(o.accelerator, [...(grouped.get(o.accelerator) ?? []), o])
  }
  const models = [...grouped].map(([name, own]): Model => {
    const entry = catalog.get(name)
    const perAccelerator = own.flatMap((o) => buysOf(o, w.allocation).map((b) => ({ ...b, price: b.price / Math.max(1, o.accelerator_count) })))
    return {
      name,
      label: labelOf(name),
      vram: entry?.vram ?? Math.max(0, ...own.map((o) => o.vram ?? 0)),
      make: entry ? `${entry.manufacturer} ${entry.architecture}` : 'unclassified',
      line: lineOf(name),
      from: cheapest(perAccelerator, w.allocation)?.price ?? 0,
    }
  })

  const want = catalog.get(w.accel)?.vram ?? offers.find((o) => o.accelerator === w.accel)?.vram ?? 0
  const accel =
    !models.length || models.some((model) => model.name === w.accel)
      ? w.accel
      : models.reduce((best, model) => (Math.abs(model.vram - want) < Math.abs(best.vram - want) ? model : best)).name

  const fits = (o: Offer) =>
    (!w.count || (o.accelerator === accel && o.accelerator_count === w.count)) &&
    (w.cpus === null || o.cpus >= w.cpus) &&
    (w.memory === null || o.memory_gb >= w.memory)
  const buys = offers.filter((o) => inAccounts(o) && fits(o)).flatMap((o) => buysOf(o, w.allocation))

  return { models, accel, buys, choice: cheapest(buys, w.allocation), sold: new Map([...sold].map(([id, names]) => [id, names.size])) }
}

/** `market._buys`: an offer with no spot price is not a spot offer, and asking for spot excludes it. */
function buysOf(o: Offer, allocation: Allocation): Buy[] {
  const spot: Buy[] = allocation !== 'on_demand' && o.spot_price != null ? [{ offer: o, market: 'spot', price: o.spot_price }] : []
  const onDemand: Buy[] = allocation !== 'spot' && o.on_demand_price != null ? [{ offer: o, market: 'on_demand', price: o.on_demand_price }] : []
  return [...spot, ...onDemand]
}

/** `market._cheapest`: `spot_if_available` prefers the spot market; every other allocation prefers the price. */
function cheapest(buys: readonly Buy[], allocation: Allocation): Buy | null {
  const spot = allocation === 'spot_if_available' ? buys.filter((b) => b.market === 'spot') : []
  return (spot.length ? spot : buys).reduce<Buy | null>((best, b) => (best && best.price <= b.price ? best : b), null)
}

/** `rtx-4070tisuper` is what the catalog keys a card by; RTX 4070 Ti Super is what a person calls it. */
function labelOf(name: string): string {
  const known = LABELS[name]
  if (known) return known
  return name
    .split('-')
    .map((part) =>
      (part.match(/ti(?=super|$)|super|ada|laptop|maxq|wk|sff|xtx|xt|\d+|[a-z]/g) ?? []).reduce((text, token) => {
        const word = WORDS[token]
        if (word) return `${text} ${word}`
        if (/\d/.test(token) && /[A-Z]{3}$/.test(text)) return `${text} ${token}`
        return text + token.toUpperCase()
      }, ''),
    )
    .join(' ')
    .trim()
}

/** Who a card is sold to, read off its name: the catalog records who makes a card, not who it is for. */
function lineOf(name: string): Line {
  if (DATACENTER.has(name) || /^(mi\d|gaudi|inferentia|trainium|instinct|tpu)/.test(name)) return 'datacenter'
  if (/^(rtx-a|rtx-pro|quadro|radeon-pro)/.test(name) || name.includes('ada')) return 'pro'
  if (/^(gtx|rtx|rx-|titan)/.test(name)) return 'consumer'
  return 'other'
}

const toggled = (list: readonly string[], item: string): readonly string[] => (list.includes(item) ? list.filter((x) => x !== item) : [...list, item])

type StepProps = { w: Draft; patch: (p: Partial<Draft>) => void; m: Market }

function Need({ w, patch, m, shelf, accounts, tone }: StepProps & { shelf: Shelf; accounts: readonly Provider[]; tone: (providerId: string) => string }) {
  const chosen = accounts.filter((p) => w.providers.includes(p.id)).map((p) => p.name)
  return (
    <div style={{ ...STACK, gap: 16 }}>
      <div className="r2" style={DUO}>
        <div className="field">
          <label htmlFor="wiz-name">Name</label>
          <input id="wiz-name" value={w.name} placeholder="llama-3-sft" onChange={(e) => patch({ name: e.target.value })} />
        </div>
        <div>
          <div className="cap">Price to take</div>
          <div style={{ marginTop: 7 }}>
            <Pick value={w.allocation} options={ALLOCS} onChange={(allocation) => patch({ allocation })} />
          </div>
        </div>
      </div>

      <div className="r2" style={DUO}>
        <div className="field">
          <label htmlFor="wiz-accel">
            Accelerator{' '}
            <span className="faint">
              · {m.models.length} from {chosen.length ? chosen.join(', ') : 'your accounts'}
            </span>
          </label>
          <select
            id="wiz-accel"
            disabled={shelf.state !== 'ready'}
            value={w.count ? m.accel : ''}
            onChange={(e) => patch(e.target.value ? { accel: e.target.value, count: w.count || 1 } : { count: 0 })}
          >
            <option value="">None</option>
            {LINES.map(([line, title]) => {
              const list = m.models.filter((model) => model.line === line).sort((a, b) => b.vram - a.vram || a.label.localeCompare(b.label))
              return list.length ? (
                <optgroup key={line} label={title}>
                  {list.map((model) => (
                    <option key={model.name} value={model.name}>
                      {`${model.label} · ${model.vram ? `${Math.round(model.vram)}GB` : '—'} · ${model.make} · from ${money(model.from, model.from < 1 ? 3 : 2)}/accelerator·h`}
                    </option>
                  ))}
                </optgroup>
              ) : null
            })}
          </select>
        </div>
        <div>
          <div className="cap">Accounts</div>
          <div className="chips" style={{ marginTop: 7 }}>
            <Chip pressed={!w.providers.length} onClick={() => patch({ providers: [] })}>
              any
            </Chip>
            {accounts.map((p) => (
              <Chip key={p.id} pressed={w.providers.includes(p.id)} onClick={() => patch({ providers: toggled(w.providers, p.id) })}>
                <i className="dot" style={{ background: tone(p.id) }} />
                {p.name}
                <span className="mono faint" style={{ fontSize: 10 }}>
                  {m.sold.get(p.id) ?? 0}
                </span>
              </Chip>
            ))}
          </div>
        </div>
      </div>

      <div className="r2" style={DUO}>
        <div>
          <div className="cap">Each node</div>
          <div style={TRIO}>
            <Num id="wiz-count" label="Accelerators" min={0} value={w.count} onChange={(count) => patch({ count })} />
            <Num id="wiz-cpus" label="vCPUs" placeholder="any" optional value={w.cpus} onChange={(cpus) => patch({ cpus })} />
            <Num id="wiz-mem" label="RAM (GB)" placeholder="any" optional value={w.memory} onChange={(memory) => patch({ memory })} />
          </div>
        </div>
        <div>
          <div className="cap">How many nodes</div>
          <div style={TRIO}>
            <Num id="wiz-min" label="Floor" min={0} placeholder={String(w.initial)} optional value={w.min} onChange={(min) => patch({ min })} />
            <Num id="wiz-max" label="Ceiling" placeholder="none" optional value={w.max} onChange={(max) => patch({ max })} />
            <Num id="wiz-initial" label="Start at" min={0} value={w.initial} onChange={(initial) => patch({ initial })} />
          </div>
        </div>
      </div>

      <div style={{ marginTop: 'auto' }}>
        <Machine shelf={shelf} choice={m.choice} tone={tone} />
      </div>
    </div>
  )
}

function Runs({ w, patch, m }: StepProps) {
  const kinds = [...new Set(m.buys.map((b) => b.offer.kind))]
  const containers = kinds.filter((k) => CONTAINERS.has(k))
  return (
    <div style={{ ...STACK, gap: 14 }}>
      <div className="r2" style={{ ...DUO, gap: 10 }}>
        {containers.length ? (
          <div className="field">
            <label htmlFor="wiz-base">
              Base image{containers.length < kinds.length ? <span className="faint"> · {containers.join(', ')} only</span> : null}
            </label>
            <input id="wiz-base" value={w.base} placeholder="provider default" onChange={(e) => patch({ base: e.target.value })} />
          </div>
        ) : null}
        <div className="field">
          <label htmlFor="wiz-python">Python</label>
          <input id="wiz-python" value={w.python} onChange={(e) => patch({ python: e.target.value })} />
        </div>
      </div>
      <Packages pip={w.pip} onChange={(pip) => patch({ pip })} />
      <div>
        <div className="cap">Plugins</div>
        <div className="chips" style={{ marginTop: 7 }}>
          {PLUGINS.map((k) => (
            <Chip key={k} pressed={w.plugins.includes(k)} onClick={() => patch({ plugins: toggled(w.plugins, k) })}>
              {k}
              {COLLECTIVE.has(k) ? (
                <span className="faint" style={{ fontSize: 9.5 }}>
                  collective
                </span>
              ) : null}
            </Chip>
          ))}
        </div>
      </div>
      <div className="r2" style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 10, alignItems: 'end' }}>
        <div>
          <div className="cap">Executor</div>
          <div style={{ marginTop: 6 }}>
            <Pick
              value={w.executor}
              options={(['thread', 'process', 'loky'] as const).map((v) => [v, v] as const)}
              onChange={(executor) => patch({ executor })}
            />
          </div>
        </div>
        <Num id="wiz-conc" label="Workers per node" value={w.concurrency} onChange={(concurrency) => patch({ concurrency })} />
        <Num id="wiz-ttl" label="Dead-switch TTL (s)" value={w.ttl} onChange={(ttl) => patch({ ttl })} />
        {w.executor === 'process' ? (
          <label className="row" style={{ gap: 7, cursor: 'pointer', height: 36 }}>
            <input type="checkbox" checked={w.reuse} onChange={(e) => patch({ reuse: e.target.checked })} />
            <span>Reuse processes</span>
          </label>
        ) : null}
      </div>
    </div>
  )
}

function Review({ w, m, shelf, tone }: Omit<StepProps, 'patch'> & { shelf: Shelf; tone: (providerId: string) => string }) {
  const containers = m.buys.some((b) => CONTAINERS.has(b.offer.kind))
  const none = <span className="faint">none</span>
  const any = <span className="faint">any</span>
  return (
    <div style={{ display: 'grid', gap: 12 }}>
      <div className="row wrap" style={{ gap: 26 }}>
        <div className="gauge-r">
          <b>
            {w.initial}
            <small>{w.count ? `× ${w.count}× ${labelOf(m.accel)}` : '× no accelerator'}</small>
          </b>
          <span>shape</span>
        </div>
        <Meter value={m.choice ? money(m.choice.price) : '—'} label="per node" />
        <Meter value={m.choice ? money(m.choice.price * w.initial) : '—'} label="per hour" />
      </div>
      <Machine shelf={shelf} choice={m.choice} tone={tone} />
      <div>
        <div className="cap">Spec</div>
        <div className="specgrid" style={{ gridAutoFlow: 'dense' }}>
          <Item label="floor">{w.min ?? w.initial}</Item>
          <Item label="ceiling">{w.max ?? none}</Item>
          <Item label="start at">{w.initial}</Item>
          <Item label="vcpus">{w.cpus ?? any}</Item>
          <Item label="ram">{w.memory ? `${w.memory} GB` : any}</Item>
          <Item label="price to take">{ALLOCS.find(([v]) => v === w.allocation)?.[1]}</Item>
          {containers ? (
            <Item label="container" wide={w.base.length > 28 ? 'wide3' : 'wide'}>
              {w.base || <span className="faint">provider default</span>}
            </Item>
          ) : null}
          <Item label="python">{w.python || none}</Item>
          <Item label="pip" wide="wide">
            {w.pip.length ? <Tick text={w.pip.join(', ')} /> : none}
          </Item>
          <Item label="plugins" wide="wide3">
            {w.plugins.length ? <Tick text={w.plugins.join(', ')} /> : none}
          </Item>
          <Item label="executor" wide="wide">
            {`${w.executor} × ${w.concurrency}${w.executor === 'process' && !w.reuse ? ', one process per task' : ''}`}
          </Item>
          {w.name ? <Item label="name">{w.name}</Item> : null}
          <Item label="dead-switch">{`${w.ttl}s`}</Item>
        </div>
      </div>
    </div>
  )
}

/** The offer the daemon would buy for what is asked so far. */
function Machine({ shelf, choice, tone }: { shelf: Shelf; choice: Buy | null; tone: (providerId: string) => string }) {
  const strip = (children: ReactNode) => (
    <div className="strip" style={{ background: 'var(--sunk)', flexWrap: 'wrap', gap: 9, minHeight: 44 }}>
      {children}
    </div>
  )
  if (shelf.state === 'loading') return strip(<span className="sub">Reading the market…</span>)
  if (shelf.state === 'failed') return strip(<span className="sub">{shelf.message}</span>)
  if (!choice) return strip(<span className="sub">No machine matches. Change the accelerator count, the vCPU and RAM floors, or the accounts.</span>)
  const o = choice.offer
  const size = [
    o.accelerator && o.accelerator_count ? `${o.accelerator_count}× ${labelOf(o.accelerator)}` : '',
    o.cpus ? `${o.cpus} vCPU` : '',
    o.memory_gb ? `${Math.round(o.memory_gb)} GB` : '',
  ]
  return strip(
    <>
      <i className="dot" style={{ background: tone(o.provider_id) }} />
      <b style={{ fontWeight: 600 }}>{o.kind}</b>
      <span className="mono">{o.instance_type}</span>
      <span className="mono faint">{[...size, o.region].filter(Boolean).join(' · ')}</span>
      <span className="mono" style={{ marginLeft: 'auto' }}>
        {money(choice.price)}/h {choice.market === 'spot' ? 'spot' : 'on demand'}
      </span>
    </>,
  )
}

const Meter = ({ value, label }: { value: string; label: string }) => (
  <div className="gauge-r">
    <b>{value}</b>
    <span>{label}</span>
  </div>
)

const Item = ({ label, wide, children }: { label: string; wide?: 'wide' | 'wide3'; children: ReactNode }) => (
  <div className={wide ? `spec-i ${wide}` : 'spec-i'}>
    <span>{label}</span>
    <span className="mono">{children}</span>
  </div>
)

type NumProps = { id: string; label: string; min?: number; placeholder?: string } & (
  | { optional?: false; value: number; onChange: (value: number) => void }
  | { optional: true; value: number | null; onChange: (value: number | null) => void }
)

/**
 * A number field that can be emptied while it is retyped. An emptied required field writes nothing, rather than a
 * zero or a default that would turn "0" into "10", and shows its value again once it loses focus.
 */
function Num(props: NumProps) {
  const [draft, setDraft] = useState<string | null>(null)
  return (
    <div className="field">
      <label htmlFor={props.id}>{props.label}</label>
      <input
        id={props.id}
        type="number"
        min={props.min ?? 1}
        placeholder={props.placeholder}
        value={draft !== null && (draft === '' || Number(draft) === props.value) ? draft : (props.value ?? '')}
        onChange={(e) => {
          setDraft(e.target.value)
          if (e.target.value !== '') props.onChange(Number(e.target.value))
          else if (props.optional) props.onChange(null)
        }}
        onBlur={() => setDraft(null)}
      />
    </div>
  )
}

/** A tag field: a comma or Enter turns what was typed into a package, and Backspace on an empty field takes the last one back. */
function Packages({ pip, onChange }: { pip: readonly string[]; onChange: (pip: readonly string[]) => void }) {
  const [text, setText] = useState('')
  const box = useRef<HTMLDivElement>(null)
  const input = useRef<HTMLInputElement>(null)

  useLayoutEffect(() => {
    if (box.current) box.current.scrollLeft = box.current.scrollWidth
  }, [pip])

  /* pip names are case-insensitive, so `NumPy` after `numpy` adds nothing */
  const add = (names: readonly string[]) =>
    onChange(names.map((t) => t.trim()).reduce((list, t) => (t && !list.some((p) => p.toLowerCase() === t.toLowerCase()) ? [...list, t] : list), pip))

  return (
    <div className="field" style={{ minWidth: 0 }}>
      <label htmlFor="wiz-pip">
        Packages{pip.length ? <span className="faint"> · {pip.length}</span> : null}
      </label>
      <div className="pkgs" ref={box} onClick={() => input.current?.focus()}>
        {pip.map((p, i) => (
          <span className="pkg mono" key={p}>
            {p}
            <button aria-label={`Remove ${p}`} onMouseDown={(e) => e.preventDefault()} onClick={() => onChange(pip.filter((_, j) => j !== i))}>
              <Icon name="close" />
            </button>
          </span>
        ))}
        <input
          id="wiz-pip"
          ref={input}
          autoComplete="off"
          spellCheck={false}
          placeholder={pip.length ? '' : 'add a package'}
          value={text}
          onChange={(e) => {
            const parts = e.target.value.split(',')
            const rest = parts.pop() ?? ''
            if (parts.length) add(parts)
            setText(parts.length ? rest.trimStart() : rest)
          }}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              e.preventDefault()
              add([text])
              setText('')
            } else if (e.key === 'Backspace' && !text && pip.length) onChange(pip.slice(0, -1))
          }}
          onBlur={() => {
            if (!text.trim()) return
            add([text])
            setText('')
          }}
        />
      </div>
    </div>
  )
}
