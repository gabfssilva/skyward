import type { LogEntry, Schemas } from './client'

/* ---------- the payloads, as the daemon tags them ---------- */

export type ComputeCreatedEvent = Schemas['ComputeCreatedEvent']
export type ComputeBoundEvent = Schemas['ComputeBoundEvent']
export type ComputeAdoptedEvent = Schemas['ComputeAdoptedEvent']
export type ComputeProvisioningEvent = Schemas['ComputeProvisioningEvent']
export type ComputeReadyEvent = Schemas['ComputeReadyEvent']
export type ComputeDegradedEvent = Schemas['ComputeDegradedEvent']
export type GenerationCreatedEvent = Schemas['ComputeGenerationCreatedEvent']
export type GenerationAppliedEvent = Schemas['ComputeGenerationAppliedEvent']
export type LeaseClaimedEvent = Schemas['ComputeLeaseClaimedEvent']
export type LeaseReleasedEvent = Schemas['ComputeLeaseReleasedEvent']
export type ComputeAbandonedEvent = Schemas['ComputeAbandonedEvent']
export type ComputeDeletingEvent = Schemas['ComputeDeletingEvent']
export type ComputeDeletionFailedEvent = Schemas['ComputeDeletionFailedEvent']
export type StraysTerminatedEvent = Schemas['ComputeStraysTerminatedEvent']
export type ComputeDeletedEvent = Schemas['ComputeDeletedEvent']
export type CostEvent = Schemas['ComputeCostEvent']
export type NodeStateEvent = Schemas['NodeStateEvent']
export type ProgressEvent = Schemas['NodeProgressEvent']
export type ConsoleEvent = Schemas['NodeConsoleEvent']
export type PhaseEvent = Schemas['NodePhaseEvent']
export type MetricEvent = Schemas['NodeMetricsEvent']
export type TaskStateEvent = Schemas['TaskStateEvent']

/** Everything the stream can carry, discriminated by `type`. */
export type EventPayload =
  | ComputeCreatedEvent
  | ComputeBoundEvent
  | ComputeAdoptedEvent
  | ComputeProvisioningEvent
  | ComputeReadyEvent
  | ComputeDegradedEvent
  | GenerationCreatedEvent
  | GenerationAppliedEvent
  | LeaseClaimedEvent
  | LeaseReleasedEvent
  | ComputeAbandonedEvent
  | ComputeDeletingEvent
  | ComputeDeletionFailedEvent
  | StraysTerminatedEvent
  | ComputeDeletedEvent
  | CostEvent
  | NodeStateEvent
  | ProgressEvent
  | ConsoleEvent
  | PhaseEvent
  | MetricEvent
  | TaskStateEvent

export type EventType = EventPayload['type']

/** One message off the stream: the frame's identity plus its decoded payload. */
export type SkyEvent = {
  /** the global sequence the frame carried in `id:` */
  id: string
  /** the payload's `type` tag (`node.state`, not the finer frame name) */
  type: string
  /** the frame's `event:` name — finer than the tag for nodes and tasks */
  frame: string
  at: number
  compute?: string | null
  node?: string | null
  task?: string | null
  data: EventPayload
  /** one line of prose, the way the prototype's event log reads */
  text: string
}

export type EventHandler = (event: SkyEvent) => void
export type Subscription = { close: () => void }

const NODE_STATES = [
  'requested',
  'provisioning',
  'connecting',
  'bootstrapping',
  'ready',
  'draining',
  'lost',
  'deleting',
  'deleted',
  'failed',
] as const

const TASK_STATES = ['started', 'retrying', 'succeeded', 'failed', 'timed_out', 'indeterminate'] as const

/** Every `event:` name a frame can go out under — an EventSource has no catch-all. */
export const FRAMES: readonly string[] = [
  'compute.created',
  'compute.bound',
  'compute.adopted',
  'compute.provisioning',
  'compute.ready',
  'compute.degraded',
  'compute.generation.created',
  'compute.generation.applied',
  'compute.lease.claimed',
  'compute.lease.released',
  'compute.abandoned',
  'compute.deleting',
  'compute.deletion_failed',
  'compute.strays_terminated',
  'compute.deleted',
  'compute.cost',
  'node.progress',
  'node.console',
  'node.phase',
  'node.metrics',
  ...NODE_STATES.map((s) => `node.${s}`),
  ...TASK_STATES.map((s) => `task.${s}`),
]

export const money = (n: number): string => '$' + n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })

/** What the event log prints for one fact. */
export function describe(payload: EventPayload): string {
  switch (payload.type) {
    case 'compute.created':
      return 'created'
    case 'compute.bound':
      return `bound to ${payload.instance_type} in ${payload.region ?? 'any region'}`
    case 'compute.adopted':
      return 'adopted by this daemon'
    case 'compute.provisioning':
      return `provisioning ${payload.nodes_ready} of ${payload.nodes_total} nodes, generation ${payload.generation}`
    case 'compute.ready':
      return `${payload.nodes_ready} of ${payload.nodes_total} nodes ready, generation ${payload.generation} applied`
    case 'compute.degraded':
      return payload.error
    case 'compute.generation.created':
      return `generation ${payload.number} created`
    case 'compute.generation.applied':
      return `generation ${payload.number} applied`
    case 'compute.lease.claimed':
      return `lease claimed by ${payload.owner}`
    case 'compute.lease.released':
      return 'lease released'
    case 'compute.abandoned':
      return 'abandoned — nobody is renewing the lease'
    case 'compute.deleting':
      return `deleting, ${payload.nodes_total} nodes to release`
    case 'compute.deletion_failed':
      return payload.error
    case 'compute.strays_terminated':
      return `${payload.machines.length} stray machines terminated`
    case 'compute.deleted':
      return 'deleted'
    case 'compute.cost':
      return `${money(payload.cost)} accrued over ${payload.nodes} nodes`
    case 'node.state':
      return payload.error ? `${payload.state} — ${payload.error}` : payload.state
    case 'node.progress':
      return payload.progress
    case 'node.console':
      return payload.content
    case 'node.phase':
      return payload.error ? `${payload.phase} ${payload.event} — ${payload.error}` : `${payload.phase} ${payload.event}`
    case 'node.metrics':
      return `${payload.name} ${payload.value}`
    case 'task.state':
      return `${payload.state} on attempt ${payload.attempt}`
  }
}

const nodeOf = (payload: EventPayload): string | null => ('node' in payload ? payload.node : null)
const taskOf = (payload: EventPayload): string | null => {
  if (payload.type === 'task.state') return payload.task
  if (payload.type === 'node.console') return payload.task ?? null
  return null
}
const atOf = (payload: EventPayload): number => {
  if ('at' in payload && typeof payload.at === 'string') return Date.parse(payload.at)
  return Date.now()
}

const skyEvent = (id: string, frame: string, at: number, data: EventPayload): SkyEvent => ({
  id,
  type: data.type,
  frame,
  at,
  compute: data.compute ?? null,
  node: nodeOf(data),
  task: taskOf(data),
  data,
  text: describe(data),
})

export function decode(frame: string, id: string, raw: string): SkyEvent | null {
  let data: EventPayload
  try {
    data = JSON.parse(raw) as EventPayload
  } catch {
    return null
  }
  if (typeof data !== 'object' || data === null || !('type' in data)) return null
  return skyEvent(id, frame, atOf(data), data)
}

/** An entry of the log as the stream would have handed it over, stamped with when the daemon recorded it. */
export const recorded = (entry: LogEntry): SkyEvent => skyEvent(String(entry.sequence), entry.type, Date.parse(entry.at), entry.data)

export type SubscribeOptions = {
  compute?: string
  task?: string
  types?: readonly string[]
  /** where to resume from — a sequence the caller has already seen, or {@link HEAD}. */
  lastEventId?: string
}

/**
 * The cursor that means "nothing that already happened".
 *
 * The replay is bounded by ``sequence > cursor``, so a cursor past every row opens
 * a stream on the live tail alone. A client that has just read the snapshot wants
 * exactly that: the log holds every console line ever printed, and replaying it is
 * both useless and, at a hundred thousand frames, fatal.
 */
export const HEAD = String(Number.MAX_SAFE_INTEGER)

/** The state-carrying frames the store folds — no console, no bootstrap phases. */
export const STATE_FRAMES: readonly string[] = FRAMES.filter((f) => f !== 'node.console' && f !== 'node.phase')

/** What a compute's dock subscribes to, one compute at a time. */
export const CONSOLE_FRAMES: readonly string[] = ['node.console', 'node.phase']

const RETRY_MIN = 500
const RETRY_MAX = 10_000

const wait = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms))

/** The blocks of an SSE body, one per blank line. */
async function* blocks(body: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const reader = body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) return
    buffer += decoder.decode(value, { stream: true })
    for (;;) {
      const end = /\r?\n\r?\n/.exec(buffer)
      if (!end) break
      yield buffer.slice(0, end.index)
      buffer = buffer.slice(end.index + end[0].length)
    }
  }
}

/** One SSE block, as the fields the daemon sets. */
function read(block: string): SkyEvent | null {
  let frame = 'message'
  let id = ''
  const data: string[] = []
  for (const line of block.split(/\r?\n/)) {
    if (!line || line.startsWith(':')) continue
    const colon = line.indexOf(':')
    const field = colon < 0 ? line : line.slice(0, colon)
    const value = colon < 0 ? '' : line.slice(line[colon + 1] === ' ' ? colon + 2 : colon + 1)
    if (field === 'event') frame = value
    else if (field === 'id') id = value
    else if (field === 'data') data.push(value)
  }
  return data.length ? decode(frame, id, data.join('\n')) : null
}

/**
 * Subscribe to the daemon's event stream.
 *
 * Read with ``fetch`` rather than ``EventSource`` for one reason: ``Last-Event-ID``
 * is a request header, and an ``EventSource`` sets it itself only after it has been
 * connected once. A client that cannot name its own cursor on the *first* connection
 * has to take the whole log, which is what freezes the page.
 *
 * The cursor is remembered as the stream runs, so a reconnect resumes where the last
 * one stopped. Published frames — metrics, progress, cost — carry the last recorded
 * sequence rather than one of their own, and never move it.
 */
export function subscribe(onEvent: EventHandler, options: SubscribeOptions = {}): Subscription {
  const params = new URLSearchParams()
  if (options.compute) params.set('compute', options.compute)
  if (options.task) params.set('task', options.task)
  for (const t of options.types ?? []) params.append('types', t)
  const query = params.toString()
  const url = `/v1/events${query ? `?${query}` : ''}`

  const abort = new AbortController()
  let cursor = options.lastEventId
  let closed = false

  const advance = (id: string): void => {
    if (!/^[1-9][0-9]*$/.test(id)) return
    if (cursor === undefined || cursor === HEAD || Number(id) > Number(cursor)) cursor = id
  }

  const run = async (): Promise<void> => {
    let attempt = 0
    while (!closed) {
      try {
        const response = await fetch(url, {
          signal: abort.signal,
          cache: 'no-store',
          headers: { accept: 'text/event-stream', ...(cursor ? { 'Last-Event-ID': cursor } : {}) },
        })
        if (!response.ok || !response.body) throw new Error(`event stream: ${response.status}`)
        attempt = 0
        for await (const block of blocks(response.body)) {
          const event = read(block)
          if (!event) continue
          advance(event.id)
          onEvent(event)
        }
      } catch {
        /* a dropped stream is resumed from the cursor, not reported */
      }
      if (closed) return
      await wait(Math.min(RETRY_MAX, RETRY_MIN * 2 ** attempt++))
    }
  }

  void run()

  return {
    close: () => {
      closed = true
      abort.abort()
    },
  }
}
