import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import type { ComputeCreate, Offer, PluginRef, Spec } from '../api/client'
import { useStore } from '../state/store'
import { money } from '../state/model'
import { Icon } from '../ui/icons'
import { Scrim, CloseBtn } from './Scrim'
import { ACCELS, COLLECTIVE, PLUGINS, PROVIDER_NAMES } from './catalog'

const WSTEPS = ['What you need', 'Where to buy it', 'What runs on it', 'Review']

/** How much of the price-ordered catalog step 2 asks the daemon for. */
const OFFERS = 200

type Mode = 'fixed' | 'partial' | 'elastic'
type Allocation = 'spot' | 'on_demand' | 'spot_if_available' | 'cheapest'
type Selection = 'cheapest' | 'first'
type Executor = 'thread' | 'process' | 'loky'

type Draft = {
  step: number
  accel: string
  count: number
  mode: Mode
  initial: number
  min: number
  max: number
  allocation: Allocation
  selection: Selection
  picks: string[]
  base: string
  python: string
  pip: string
  plugins: string[]
  executor: Executor
  concurrency: number
  name: string
  ttl: number
  deleteOnExit: boolean
}

const newWizard = (offer: Offer | undefined): Draft => ({
  step: offer ? 2 : 1,
  accel: offer?.accelerator ?? 'h100',
  count: offer?.accelerator_count ?? 8,
  mode: 'fixed',
  initial: 8,
  min: 8,
  max: 16,
  allocation: 'spot_if_available',
  selection: 'cheapest',
  picks: offer ? [offer.id] : [],
  base: 'pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel',
  python: '3.12',
  pip: 'transformers, trl',
  plugins: ['torch'],
  executor: 'thread',
  concurrency: 8,
  name: '',
  ttl: 600,
  deleteOnExit: true,
})

/** The offer the wizard opens on, handed over by the market view through `openWizard`. */
let seed: Offer | undefined

export const seedWizard = (offer: Offer | undefined): void => {
  seed = offer
}

const ondemandOf = (o: Offer): number => o.on_demand_price ?? o.price ?? 0
const priceOf = (o: Offer, allocation: Allocation): number => (allocation === 'on_demand' ? ondemandOf(o) : (o.spot_price ?? ondemandOf(o)))

export function Wizard() {
  const navigate = useNavigate()
  const closeSheet = useStore((s) => s.closeSheet)
  const load = useStore((s) => s.load)
  const [w, setW] = useState<Draft>(() => newWizard(seed))
  const [opened] = useState(() => seed)
  const [fetched, setFetched] = useState<Offer[]>(() => (seed ? [seed] : []))
  const [busy, setBusy] = useState(false)
  const patch = (p: Partial<Draft>) => setW((prev) => ({ ...prev, ...p }))

  useEffect(() => {
    let live = true
    void api
      .offers({ accelerator: w.accel, min_count: w.count, sort: 'price', limit: OFFERS })
      .then((read) => {
        if (live) setFetched(read.items)
      })
      .catch(() => {
        if (live) setFetched([])
      })
    return () => {
      live = false
    }
  }, [w.accel, w.count])

  /* the offer the market opened on can sit outside the slice the daemon answered with, and a seeded pick still has to resolve */
  const offers = useMemo(() => (opened && !fetched.some((o) => o.id === opened.id) ? [opened, ...fetched] : fetched), [opened, fetched])
  const picked = useMemo(() => w.picks.map((id) => offers.find((o) => o.id === id)).filter((o): o is Offer => !!o), [w.picks, offers])
  const price = useMemo(() => {
    if (!picked.length) return null
    const chosen = w.selection === 'cheapest' ? picked.slice().sort((a, b) => priceOf(a, w.allocation) - priceOf(b, w.allocation))[0]! : picked[0]!
    const node = priceOf(chosen, w.allocation)
    return { node, offer: chosen, total: node * Number(w.initial || 1) }
  }, [picked, w.selection, w.allocation, w.initial])

  const create = async () => {
    setBusy(true)
    try {
      const specs: Spec[] = picked.map((o) => ({
        provider: { kind: o.kind, name: o.provider_name },
        accelerator: o.accelerator,
        accelerator_count: o.accelerator_count,
        cpus: o.cpus,
        memory_gb: o.memory_gb,
        region: o.region,
      }))
      const plugins: PluginRef[] = w.plugins.map((kind) => ({ kind, params: {} }))
      const payload: ComputeCreate = {
        name: w.name || null,
        spec: {
          allocation: w.allocation,
          selection: w.selection,
          delete_on_exit: w.deleteOnExit,
          desired: 'running',
          ttl: Number(w.ttl),
          nodes: {
            initial: Number(w.initial),
            min: Number(w.mode === 'fixed' ? w.initial : w.min),
            max: w.mode === 'elastic' ? Number(w.max) : null,
          },
          specs,
          plugins,
          volumes: [],
          image: {
            base: w.base,
            python: w.python,
            pip: w.pip.split(',').map((s) => s.trim()).filter(Boolean),
            apt: [],
            excludes: [],
            includes: [],
            pip_indexes: [],
            bootstrap_timeout: 900,
            skyward: 'auto',
            warm: false,
          },
          worker: { executor: w.executor, concurrency: Number(w.concurrency), buffer: 0, reuse: true },
        },
      }
      const compute = await api.createCompute(payload)
      closeSheet()
      await load()
      navigate(`/computes/${compute.id}`)
    } finally {
      setBusy(false)
    }
  }

  const next = () => {
    if (w.step === 4) {
      void create()
      return
    }
    patch({ step: w.step + 1 })
  }

  return (
    <Scrim>
      <div className="sheet" role="dialog" aria-label="New compute">
        <div className="sheet-head">
          <b>New compute</b>
          <span className="sub">{WSTEPS[w.step - 1]}</span>
          <div className="steps">
            {WSTEPS.map((s, i) => (
              <i key={s} className={i < w.step ? 'on' : ''} />
            ))}
          </div>
          <CloseBtn />
        </div>
        <div className="sheet-body">
          {w.step === 1 ? <Step1 w={w} patch={patch} /> : null}
          {w.step === 2 ? <Step2 w={w} patch={patch} offers={offers} /> : null}
          {w.step === 3 ? <Step3 w={w} patch={patch} /> : null}
          {w.step === 4 ? <Step4 w={w} picked={picked} price={price} /> : null}
        </div>
        <div className="sheet-foot">
          {w.step > 1 ? (
            <button className="btn" onClick={() => patch({ step: w.step - 1 })}>
              <Icon name="back" />
              Back
            </button>
          ) : null}
          <span className="sub">
            {price ? `${money(price.total)} per hour · ${money(price.total * 8, 0)} for an eight-hour run` : 'Pick an offer to price it'}
          </span>
          <button className="btn primary" style={{ marginLeft: 'auto' }} disabled={busy || (w.step === 2 && !w.picks.length)} onClick={next}>
            {w.step === 4 ? (
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

type StepProps = { w: Draft; patch: (p: Partial<Draft>) => void }

const ALLOCS: readonly (readonly [Allocation, string])[] = [
  ['spot_if_available', 'spot if available'],
  ['spot', 'spot only'],
  ['on_demand', 'on demand'],
  ['cheapest', 'cheapest'],
]

const MODES: readonly (readonly [Mode, string])[] = [
  ['fixed', 'fixed'],
  ['partial', 'start early'],
  ['elastic', 'elastic'],
]

function Step1({ w, patch }: StepProps) {
  return (
    <div style={{ display: 'grid', gap: 16 }}>
      <div>
        <div className="cap">Accelerator</div>
        <div className="chips" style={{ marginTop: 7 }}>
          {Object.entries(ACCELS).map(([k, a]) => (
            <button key={k} className="chip" aria-pressed={w.accel === k} style={{ height: 30 }} onClick={() => patch({ accel: k, picks: [] })}>
              {k.toUpperCase()}{' '}
              <span className="mono faint" style={{ fontSize: 10 }}>
                {a.vram}GB
              </span>
            </button>
          ))}
        </div>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }} className="r2">
        <div>
          <div className="cap">GPUs per node</div>
          <div className="pick" style={{ marginTop: 7 }}>
            {[1, 2, 4, 8].map((n) => (
              <button key={n} aria-selected={w.count === n} onClick={() => patch({ count: n, picks: [] })}>
                {n}
              </button>
            ))}
          </div>
        </div>
        <div>
          <div className="cap">Price to take</div>
          <div className="pick" style={{ marginTop: 7 }}>
            {ALLOCS.map(([v, l]) => (
              <button key={v} aria-selected={w.allocation === v} onClick={() => patch({ allocation: v })}>
                {l}
              </button>
            ))}
          </div>
        </div>
      </div>
      <div>
        <div className="cap">How many nodes</div>
        <div className="row wrap" style={{ gap: 10, margin: '8px 0 10px' }}>
          <div className="pick">
            {MODES.map(([v, l]) => (
              <button key={v} aria-selected={w.mode === v} onClick={() => patch({ mode: v, ...(v === 'fixed' ? { min: w.initial } : {}) })}>
                {l}
              </button>
            ))}
          </div>
          <div className="chips">
            {[1, 4, 8, 16, 32, 64, 128].map((n) => (
              <button
                key={n}
                className="chip"
                aria-pressed={Number(w.initial) === n}
                onClick={() => patch({ initial: n, ...(w.mode === 'fixed' ? { min: n } : {}) })}
              >
                {n}
              </button>
            ))}
          </div>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,120px)', gap: 10 }}>
          <div className="field">
            <label htmlFor="wiz-initial">Open at</label>
            <input id="wiz-initial" type="number" min={1} value={w.initial} onChange={(e) => patch({ initial: Number(e.target.value) })} />
          </div>
          {w.mode !== 'fixed' ? (
            <div className="field">
              <label htmlFor="wiz-min">Floor</label>
              <input id="wiz-min" type="number" min={1} value={w.min} onChange={(e) => patch({ min: Number(e.target.value) })} />
            </div>
          ) : null}
          {w.mode === 'elastic' ? (
            <div className="field">
              <label htmlFor="wiz-max">Ceiling</label>
              <input id="wiz-max" type="number" min={1} value={w.max} onChange={(e) => patch({ max: Number(e.target.value) })} />
            </div>
          ) : null}
        </div>
        <div className="sub" style={{ marginTop: 8 }}>
          {w.mode === 'fixed'
            ? `Work starts when all ${w.initial} nodes are ready.`
            : w.mode === 'partial'
              ? `Asks for ${w.initial}, starts working at ${w.min}; latecomers join the same generation.`
              : `Scales between ${w.min} and ${w.max} as tasks queue and nodes idle.`}
        </div>
      </div>
    </div>
  )
}

function Step2({ w, patch, offers }: StepProps & { offers: Offer[] }) {
  const sorted = offers
    .filter((o) => o.accelerator === w.accel && o.accelerator_count === w.count)
    .slice()
    .sort((a, b) => priceOf(a, w.allocation) - priceOf(b, w.allocation))
  const toggle = (id: string) => {
    const i = w.picks.indexOf(id)
    patch({ picks: i < 0 ? [...w.picks, id] : w.picks.filter((x) => x !== id) })
  }
  return (
    <div>
      <div className="combhead">
        <b>
          {sorted.length} offers match {w.count}× {w.accel.toUpperCase()}
        </b>
        <span className="sub">pick one, or several — several become a fallback chain</span>
      </div>
      {sorted.length ? (
        <div className="scroll" style={{ maxHeight: '42vh', overflowY: 'auto' }}>
          <table>
            <thead>
              <tr>
                <th />
                <th>Provider</th>
                <th>Instance</th>
                <th>Region</th>
                <th className="right">Node /h</th>
                <th className="right">{w.initial} nodes /h</th>
                <th>Available</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((o) => {
                const on = w.picks.includes(o.id)
                return (
                  <tr key={o.id} style={{ cursor: 'pointer', ...(on ? { background: 'var(--accent-soft)' } : {}) }} onClick={() => toggle(o.id)}>
                    <td style={{ width: 26 }}>
                      {on ? (
                        <span className="mono" style={{ color: 'var(--accent)' }}>
                          {w.picks.indexOf(o.id) + 1}
                        </span>
                      ) : (
                        <span className="faint">—</span>
                      )}
                    </td>
                    <td>
                      <b style={{ fontWeight: 600 }}>{o.kind}</b>
                      {o.spot_price == null ? (
                        <div className="faint" style={{ fontSize: 10 }}>
                          no spot market
                        </div>
                      ) : null}
                    </td>
                    <td className="mono">{o.instance_type}</td>
                    <td className="mono faint">{o.region}</td>
                    <td className="right mono">
                      <b>{money(priceOf(o, w.allocation))}</b>
                    </td>
                    <td className="right mono faint">{money(priceOf(o, w.allocation) * Number(w.initial || 1))}</td>
                    <td className="mono faint">{o.available ? o.available : <span style={{ color: 'var(--bad)' }}>none</span>}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="empty">
          <Icon name="market" />
          <b>No offer matches that shape</b>
          <span>Try another GPU count, or fetch fresh offers.</span>
        </div>
      )}
      {w.picks.length > 1 ? (
        <div className="strip" style={{ marginTop: 10 }}>
          <span className="sub">With {w.picks.length} specs, the daemon takes</span>
          <div className="pick">
            {([['cheapest', 'the cheapest'], ['first', 'the first that answers']] as const).map(([v, l]) => (
              <button key={v} aria-selected={w.selection === v} onClick={() => patch({ selection: v })}>
                {l}
              </button>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  )
}

function Step3({ w, patch }: StepProps) {
  const togglePlugin = (k: string) => {
    const i = w.plugins.indexOf(k)
    patch({ plugins: i < 0 ? [...w.plugins, k] : w.plugins.filter((x) => x !== k) })
  }
  return (
    <div style={{ display: 'grid', gap: 14 }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }} className="r2">
        <div className="field">
          <label htmlFor="wiz-base">Base image</label>
          <input id="wiz-base" value={w.base} onChange={(e) => patch({ base: e.target.value })} />
        </div>
        <div className="field">
          <label htmlFor="wiz-python">Python</label>
          <input id="wiz-python" value={w.python} onChange={(e) => patch({ python: e.target.value })} />
        </div>
      </div>
      <div className="field">
        <label htmlFor="wiz-pip">Packages</label>
        <input id="wiz-pip" value={w.pip} onChange={(e) => patch({ pip: e.target.value })} />
      </div>
      <div>
        <div className="cap">Plugins — each shapes the image, the bootstrap and every task</div>
        <div className="chips" style={{ marginTop: 7 }}>
          {PLUGINS.map((k) => (
            <button key={k} className="chip" aria-pressed={w.plugins.includes(k)} onClick={() => togglePlugin(k)}>
              {k}
              {COLLECTIVE.has(k) ? (
                <>
                  {' '}
                  <span className="faint" style={{ fontSize: 9.5 }}>
                    collective
                  </span>
                </>
              ) : null}
            </button>
          ))}
        </div>
        {w.plugins.some((k) => COLLECTIVE.has(k)) ? (
          <div className="sub" style={{ marginTop: 6 }}>
            A collective plugin freezes the world: the reconciler refuses to resize a compute holding one.
          </div>
        ) : null}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 10, alignItems: 'end' }} className="r2">
        <div>
          <div className="cap">Executor</div>
          <div className="pick" style={{ marginTop: 6 }}>
            {(['thread', 'process', 'loky'] as const).map((v) => (
              <button key={v} aria-selected={w.executor === v} onClick={() => patch({ executor: v })}>
                {v}
              </button>
            ))}
          </div>
        </div>
        <div className="field">
          <label htmlFor="wiz-conc">Workers per node</label>
          <input id="wiz-conc" type="number" min={1} value={w.concurrency} onChange={(e) => patch({ concurrency: Number(e.target.value) })} />
        </div>
        <div className="field">
          <label htmlFor="wiz-name">Name</label>
          <input id="wiz-name" value={w.name} placeholder="llama-3-sft" onChange={(e) => patch({ name: e.target.value })} />
        </div>
        <div className="field">
          <label htmlFor="wiz-ttl">Dead-switch TTL (s)</label>
          <input id="wiz-ttl" type="number" value={w.ttl} onChange={(e) => patch({ ttl: Number(e.target.value) })} />
        </div>
      </div>
      <label className="row" style={{ gap: 7, cursor: 'pointer' }}>
        <input type="checkbox" checked={w.deleteOnExit} onChange={(e) => patch({ deleteOnExit: e.target.checked })} />
        <span>
          Delete the machines when the block exits
          <span className="sub"> — off keeps the compute for another process to attach to</span>
        </span>
      </label>
    </div>
  )
}

const Meter = ({ value, label }: { value: string; label: string }) => (
  <div className="gauge-r">
    <b>{value}</b>
    <span>{label}</span>
  </div>
)

function snippet(w: Draft, picked: readonly Offer[]): string {
  const nodes = w.mode === 'fixed' ? String(w.initial) : w.mode === 'elastic' ? `(${w.min}, ${w.max})` : `sky.Nodes(initial=${w.initial}, min=${w.min})`
  const pip = w.pip.split(',').map((s) => s.trim()).filter(Boolean)
  return [
    'with sky.Compute(',
    ...picked.map(
      (o) =>
        `    sky.Spec(provider=sky.${PROVIDER_NAMES[o.kind] ?? o.kind}(), accelerator=sky.accelerators.${w.accel.toUpperCase()}(${w.count > 1 ? w.count : ''}), region="${o.region}"),`,
    ),
    `    nodes=${nodes},`,
    `    allocation="${w.allocation}",`,
    picked.length > 1 ? `    selection="${w.selection}",` : null,
    `    image=sky.Image(base="${w.base}", python="${w.python}"${pip.length ? `, pip=${JSON.stringify(pip)}` : ''}),`,
    w.plugins.length ? `    plugins=[${w.plugins.map((k) => `sky.plugins.${k[0]!.toUpperCase() + k.slice(1)}()`).join(', ')}],` : null,
    `    executor=sky.Executor(type="${w.executor}", concurrency=${w.concurrency}),`,
    w.name ? `    name="${w.name}",` : null,
    `    delete_on_exit=${w.deleteOnExit ? 'True' : 'False'},`,
    ') as pool:',
    '    result = train(data) @ pool',
  ]
    .filter((line): line is string => line !== null)
    .join('\n')
}

function Step4({ w, picked, price }: { w: Draft; picked: readonly Offer[]; price: { node: number; total: number } | null }) {
  return (
    <div style={{ display: 'grid', gap: 12 }}>
      <div className="row wrap" style={{ gap: 26 }}>
        <div className="gauge-r">
          <b>
            {w.initial}
            <small>
              × {w.count}× {w.accel.toUpperCase()}
            </small>
          </b>
          <span>shape</span>
        </div>
        <Meter value={price ? money(price.node) : '—'} label="per node" />
        <Meter value={price ? money(price.total) : '—'} label="per hour" />
        <Meter value={price ? money(price.total * 8, 0) : '—'} label="eight-hour run" />
      </div>
      <div className="strip" style={{ flexDirection: 'column', alignItems: 'stretch', gap: 5 }}>
        {picked.map((o, i) => (
          <div className="row" style={{ gap: 9 }} key={o.id}>
            <span className="mono faint">{i + 1}</span>
            <b style={{ fontWeight: 600 }}>{o.kind}</b>
            <span className="mono faint">
              {o.instance_type} · {o.region}
            </span>
            <span className="mono" style={{ marginLeft: 'auto' }}>
              {money(w.allocation === 'on_demand' ? ondemandOf(o) : (o.spot_price ?? ondemandOf(o)))}/h
            </span>
          </div>
        ))}
      </div>
      <div>
        <div className="cap">The same compute, written by hand</div>
        <pre className="term" style={{ margin: '8px 0 0', padding: '13px 15px', overflow: 'auto' }}>
          <code>{snippet(w, picked)}</code>
        </pre>
      </div>
    </div>
  )
}
