import type { EngineInterface, HttpResponse, Register, RenderElement, Timer } from 'claude-code'

const DAEMON = 'http://127.0.0.1:17590'
const PANE_ID = 'skyward'
const REFRESH_MS = 3000
const LOG_LINES = 20
const SPARK_STEP_MS = 30_000
const SPARK_BUCKETS = 20
const SPARK_GLYPHS = '▁▂▃▄▅▆▇█'
const NO_SAMPLES: readonly null[] = Array.from({ length: SPARK_BUCKETS }, () => null)
const NODES_PER_PAGE = 4
const TASKS_PER_PAGE = 5
const DELETE_ATTEMPTS = 5
const NAME_WIDTH = 16
const MARQUEE_MS = 250
const MARQUEE_PAUSE_FRAMES = 8

type Compute = {
  id: string
  name: string | null
  state: string
  nodesReady: number
  nodesTotal: number
  provider: string
  accelerator: string | null
  acceleratorCount: number
  price: number | null
  createdAt: number
  ended: boolean
  cost: number | null
  lastError: string | null
}

type Node = {
  id: string
  rank: number
  state: string
  address: string | null
  machine: string | null
  market: string | null
  price: number | null
  launchedAt: number | null
  terminatedAt: number | null
  lastError: string | null
}

type LogLine = { at: number; node: string | null; type: string; text: string; error: boolean }

// `cursors` is the way back from the newest page: the log only pages toward older entries.
type LogFilter = { node: string | null; task: string | null; term: string; cursors: readonly string[] }
type LogPage = { path: string; lines: LogLine[]; next: string | null }

// `node`, `startedAt` and `error` come from the latest attempt.
type Task = {
  id: string
  sha: string
  name: string | null
  state: string
  node: string | null
  attempts: number
  submittedAt: number
  startedAt: number | null
  finishedAt: number | null
  error: string | null
}

type TaskOrder = 'state' | 'submitted' | 'finished'
// Tasks also page forward only, so `cursors` is the way back to the first page.
type TaskQuery = { order: TaskOrder; fn: string | null; cursors: readonly string[] }
type TaskPage = { items: Task[]; total: number | null; next: string | null; counts: Map<string, number> }

type Detail = {
  compute: Compute
  nodes: Node[]
  page: number
  pageCount: number
  latest: Map<string, number>
  sparks: Map<string, (number | null)[]>
  tasks: TaskPage
  log: LogPage
}

type DetailView = { kind: 'detail'; id: string; page: number; expanded: ReadonlySet<string>; tasks: TaskQuery; log: LogFilter }
type View = { kind: 'list' } | DetailView

type Removal = { id: string; step: 'confirm' } | { id: string; step: 'sending' } | { id: string; step: 'failed'; reason: string }

type Snapshot =
  | { kind: 'loading' }
  | { kind: 'down'; reason: string }
  | { kind: 'list'; live: Compute[]; recent: Compute[]; at: number }
  | { kind: 'detail'; detail: Detail; at: number }

const STATE_COLORS = new Map([
  ['requested', 'yellow'],
  ['provisioning', 'yellow'],
  ['connecting', 'yellow'],
  ['bootstrapping', 'yellow'],
  ['ready', 'green'],
  ['degraded', 'red'],
  ['draining', 'magenta'],
  ['deleting', 'magenta'],
])

// Dark enough for white text on light and dark themes; no pure green, red or yellow, which the states use.
const RANK_COLORS = ['#3b6fd8', '#8e5bd6', '#1f8f8f', '#c7702a', '#c2427a', '#6b8e23', '#546e7a', '#8d6e63']
const RANK_WIDTH = 5

const TASK_ORDERS: readonly TaskOrder[] = ['state', 'submitted', 'finished']
// The rest (cancelled, timed_out, indeterminate) is shown as what the total leaves over.
const COUNTED_TASK_STATES = ['running', 'queued', 'succeeded', 'failed']
const TASK_GLYPHS = new Map([
  ['running', '●'],
  ['queued', '◌'],
  ['succeeded', '✓'],
  ['failed', '✗'],
  ['timed_out', '✗'],
  ['cancelled', '○'],
  ['indeterminate', '?'],
])
const TASK_COLORS = new Map([
  ['running', 'green'],
  ['failed', 'red'],
  ['timed_out', 'red'],
  ['indeterminate', 'magenta'],
])

const STATE_GLYPHS = new Map([
  ['requested', '◌'],
  ['provisioning', '◐'],
  ['ready', '●'],
  ['degraded', '●'],
  ['deleting', '◑'],
  ['deleted', '○'],
])

const at = (value: unknown, ...path: string[]): unknown =>
  path.reduce<unknown>((current, key) => {
    if (typeof current !== 'object' || current === null) return undefined
    const field: unknown = Object.getOwnPropertyDescriptor(current, key)?.value
    return field
  }, value)
const str = (value: unknown): string | undefined => (typeof value === 'string' ? value : undefined)
const num = (value: unknown): number | undefined => (typeof value === 'number' ? value : undefined)
const list = (value: unknown): unknown[] => (Array.isArray(value) ? value : [])
const time = (value: unknown): number | null => {
  const ms = Date.parse(str(value) ?? '')
  return Number.isNaN(ms) ? null : ms
}
const items = <T,>(body: unknown, parse: (item: unknown) => T | undefined): T[] =>
  list(at(body, 'items')).flatMap((item) => {
    const parsed = parse(item)
    return parsed === undefined ? [] : [parsed]
  })
const metricKey = (node: string, name: string): string => `${node}/${name}`

function parseCompute(item: unknown): Compute | undefined {
  const id = str(at(item, 'id'))
  const state = str(at(item, 'status', 'state'))
  const nodesReady = num(at(item, 'status', 'nodes_ready'))
  const nodesTotal = num(at(item, 'status', 'nodes_total'))
  const createdAt = time(at(item, 'created_at'))
  if (id === undefined || state === undefined || nodesReady === undefined || nodesTotal === undefined || createdAt === null) {
    return undefined
  }
  const spec = at(item, 'spec', 'specs', '0')
  return {
    id,
    state,
    nodesReady,
    nodesTotal,
    createdAt,
    name: str(at(item, 'name')) ?? null,
    provider: str(at(item, 'offer', 'provider_name')) ?? str(at(spec, 'provider', 'name')) ?? '?',
    accelerator: str(at(item, 'offer', 'accelerator')) ?? str(at(spec, 'accelerator')) ?? null,
    acceleratorCount: num(at(item, 'offer', 'accelerator_count')) ?? num(at(spec, 'accelerator_count')) ?? 1,
    price: num(at(item, 'offer', 'price')) ?? null,
    ended: time(at(item, 'ended', 'at')) !== null,
    cost: num(at(item, 'ended', 'cost')) ?? null,
    lastError: str(at(item, 'status', 'last_error', 'message')) ?? null,
  }
}

function parseNode(item: unknown): Node | undefined {
  const id = str(at(item, 'id'))
  const rank = num(at(item, 'rank'))
  const state = str(at(item, 'state'))
  if (id === undefined || rank === undefined || state === undefined) return undefined
  return {
    id,
    rank,
    state,
    address: str(at(item, 'address')) ?? null,
    machine: str(at(item, 'machine')) ?? null,
    market: str(at(item, 'market')) ?? null,
    price: num(at(item, 'price_per_hour')) ?? null,
    launchedAt: time(at(item, 'launched_at')),
    terminatedAt: time(at(item, 'terminated_at')),
    lastError: str(at(item, 'last_error', 'message')) ?? null,
  }
}

function parseLogLine(item: unknown): LogLine | undefined {
  const type = str(at(item, 'type'))
  const when = time(at(item, 'at'))
  if (type === undefined || when === null) return undefined
  const data = at(item, 'data')
  const error = str(at(data, 'error'))
  const text =
    type === 'node.console'
      ? str(at(data, 'content')) ?? ''
      : type === 'node.phase'
        ? `${str(at(data, 'phase')) ?? '?'} ${str(at(data, 'event')) ?? '?'}`
        : type
  return {
    at: when,
    type,
    node: str(at(data, 'node')) ?? null,
    text: error === undefined ? text : `${text}: ${error}`,
    error: error !== undefined,
  }
}

function parseTask(item: unknown): Task | undefined {
  const id = str(at(item, 'id'))
  const sha = str(at(item, 'function'))
  const state = str(at(item, 'state'))
  const submittedAt = time(at(item, 'submitted_at'))
  if (id === undefined || sha === undefined || state === undefined || submittedAt === null) return undefined
  const executions = list(at(item, 'executions'))
  const latest = executions.at(-1)
  return {
    id,
    sha,
    state,
    submittedAt,
    name: null,
    node: str(at(latest, 'node_id')) ?? null,
    attempts: executions.length,
    startedAt: time(at(latest, 'started_at')),
    finishedAt: time(at(item, 'finished_at')),
    error: str(at(latest, 'error', 'message')) ?? null,
  }
}

const age = (ms: number): string => {
  if (ms < 60_000) return `${Math.max(0, Math.floor(ms / 1000))}s`
  const minutes = Math.floor(ms / 60_000)
  if (minutes < 60) return `${minutes}m`
  const hours = Math.floor(minutes / 60)
  return hours < 48 ? `${hours}h` : `${Math.floor(hours / 24)}d`
}
const price = (value: number | null): string => (value === null ? '-' : `$${value.toFixed(2)}/h`)
const gib = (mb: number): string => (mb / 1024).toFixed(1)
const rate = (kbps: number): string => (kbps >= 1000 ? `${(kbps / 1000).toFixed(1)}Mb/s` : `${Math.round(kbps)}kb/s`)
const spark = (buckets: (number | null)[]): string =>
  buckets
    .map((value) => (value === null ? ' ' : SPARK_GLYPHS.charAt(Math.max(0, Math.min(7, Math.floor((value / 100) * 8))))))
    .join('')
const clock = (ms: number): string => new Date(ms).toTimeString().slice(0, 8)
const money = (value: number | null): string => (value === null ? '—' : value.toFixed(2))
const gpu = (c: Compute): string =>
  c.accelerator === null ? '—' : c.acceleratorCount > 1 ? `${c.acceleratorCount}×${c.accelerator}` : c.accelerator
const deletable = (c: Compute): boolean => !c.ended && c.state !== 'deleting' && c.state !== 'deleted'
const hourly = (c: Compute): number | null => (c.ended || c.price === null ? null : c.price * c.nodesTotal)
const sum = (values: (number | null)[]): number | null =>
  values.reduce<number | null>((total, value) => (total === null || value === null ? null : total + value), 0)

// Pauses at both ends, so a name reads from its start before it scrolls and rests on its end before jumping back.
const marquee = (text: string, width: number, now: number): string => {
  const overflow = text.length - width
  if (overflow <= 0) return text
  const frame = Math.floor(now / MARQUEE_MS) % (overflow + 2 * MARQUEE_PAUSE_FRAMES)
  const offset = Math.min(overflow, Math.max(0, frame - MARQUEE_PAUSE_FRAMES))
  return text.slice(offset, offset + width)
}

type Column = {
  title: string
  width: number
  right: boolean
  // Lower ranks stay longest when the pane is too narrow for every column.
  rank: number
  cell: (c: Compute) => { text: string; color?: string }
  total?: (live: Compute[]) => string
}

const COLUMNS: Column[] = [
  {
    title: 'STATE',
    // A glyph and up to 100/100 nodes.
    width: 9,
    right: false,
    rank: 1,
    cell: (c) => ({
      text: c.ended ? STATE_GLYPHS.get('deleted') ?? '○' : `${STATE_GLYPHS.get(c.state) ?? '●'} ${c.nodesReady}/${c.nodesTotal}`,
      color: STATE_COLORS.get(c.state),
    }),
  },
  { title: 'GPU', width: 10, right: false, rank: 2, cell: (c) => ({ text: gpu(c) }) },
  { title: 'PROVIDER', width: 9, right: false, rank: 6, cell: (c) => ({ text: c.provider }) },
  { title: '$/H', width: 7, right: true, rank: 4, cell: (c) => ({ text: money(hourly(c)) }), total: (live) => money(sum(live.map(hourly))) },
  { title: 'COST', width: 7, right: true, rank: 3, cell: (c) => ({ text: money(c.cost) }), total: (live) => money(sum(live.map((c) => c.cost))) },
  { title: 'AGE', width: 4, right: true, rank: 5, cell: (c) => ({ text: age(Date.now() - c.createdAt) }) },
]

const TABLE_GAP = 2

const fitColumns = (columns: number): Column[] => {
  const kept = new Set<Column>()
  let used = NAME_WIDTH
  for (const column of [...COLUMNS].sort((a, b) => a.rank - b.rank)) {
    if (used + TABLE_GAP + column.width > columns) break
    used += TABLE_GAP + column.width
    kept.add(column)
  }
  return COLUMNS.filter((column) => kept.has(column))
}

// The daemon refuses with `{code, message}`; a body without a message is shown as it came.
function refusal(response: HttpResponse): Error {
  let body: unknown
  try {
    body = JSON.parse(response.text)
  } catch {
    body = undefined
  }
  return new Error(`HTTP ${response.status} ${str(at(body, 'message')) ?? response.text.slice(0, 200)}`)
}

async function getJson($: EngineInterface, path: string): Promise<unknown> {
  const response = await $.http.fetch(`${DAEMON}${path}`)
  if (!response.ok) throw refusal(response)
  const body: unknown = JSON.parse(response.text)
  return body
}

async function loadList($: EngineInterface): Promise<Snapshot> {
  const [live, recent] = await Promise.all([getJson($, '/v1/computes?live=true'), getJson($, '/v1/computes?live=false&limit=5')])
  return { kind: 'list', live: items(live, parseCompute), recent: items(recent, parseCompute), at: Date.now() }
}

const logPath = (id: string, filter: LogFilter): string => {
  const cursor = filter.cursors.at(-1)
  return [
    `/v1/events/log?compute=${id}&limit=${LOG_LINES}`,
    filter.node === null ? '' : `&node=${filter.node}`,
    filter.task === null ? '' : `&task=${filter.task}`,
    filter.term === '' ? '' : `&contains=${encodeURIComponent(filter.term)}`,
    cursor === undefined ? '' : `&cursor=${cursor}`,
  ].join('')
}

async function loadLog($: EngineInterface, path: string): Promise<LogPage> {
  const body = await getJson($, path)
  return { path, lines: items(body, parseLogLine), next: str(at(body, 'next_cursor')) ?? null }
}

// `names` outlives the load: a function's sha names its code, so the name found for it never changes.
async function loadTasks($: EngineInterface, id: string, query: TaskQuery, names: Map<string, string | null>): Promise<TaskPage> {
  const filter = `/v1/tasks?compute=${id}${query.fn === null ? '' : `&function=${encodeURIComponent(query.fn)}`}`
  const cursor = query.cursors.at(-1)
  const [pageBody, ...countBodies] = await Promise.all([
    getJson($, `${filter}&limit=${TASKS_PER_PAGE}&order=${query.order}${cursor === undefined ? '' : `&cursor=${cursor}`}`),
    ...COUNTED_TASK_STATES.map((state) => getJson($, `${filter}&state=${state}&limit=1`)),
  ])
  const parsed = items(pageBody, parseTask)
  const unnamed = [...new Set(parsed.map((t) => t.sha))].filter((sha) => !names.has(sha))
  const found = await Promise.all(
    unnamed.map(async (sha): Promise<[string, string | null]> => [sha, str(at(await getJson($, `/v1/functions/${sha}`), 'name')) ?? null]),
  )
  found.forEach(([sha, name]) => names.set(sha, name))
  const total = num(at(pageBody, 'total')) ?? null
  // A last page exactly TASKS_PER_PAGE long still carries a cursor, and it leads to an empty page.
  const last = total !== null && query.cursors.length * TASKS_PER_PAGE + parsed.length >= total
  return {
    items: parsed.map((t) => ({ ...t, name: names.get(t.sha) ?? null })),
    total,
    next: last ? null : str(at(pageBody, 'next_cursor')) ?? null,
    counts: new Map(COUNTED_TASK_STATES.map((state, i) => [state, num(at(countBodies[i], 'total')) ?? 0])),
  }
}

async function loadDetail($: EngineInterface, target: DetailView, shown: LogPage | undefined, names: Map<string, string | null>): Promise<Snapshot> {
  const { id, page: requestedPage, expanded } = target
  const start = (Math.floor(Date.now() / SPARK_STEP_MS) - (SPARK_BUCKETS - 1)) * SPARK_STEP_MS
  const base = `/v1/computes/${id}`
  const path = logPath(id, target.log)
  // The log is append-only, so a page behind a cursor never changes and only the newest page is fetched again.
  const unchanged = target.log.cursors.length > 0 && shown?.path === path ? shown : undefined
  const [computeBody, nodesBody, tasks, log] = await Promise.all([
    getJson($, base),
    getJson($, `${base}/nodes`),
    loadTasks($, id, target.tasks, names),
    unchanged ?? loadLog($, path),
  ])
  const compute = parseCompute(computeBody)
  if (compute === undefined) throw new Error(`unexpected compute payload for ${id}`)

  // The nodes endpoint has no cursor, so the page is cut here; latest values are asked for that page's nodes only,
  // and the sparkline series for the expanded ones among them.
  const nodes = items(nodesBody, parseNode).sort((a, b) => Number(a.terminatedAt !== null) - Number(b.terminatedAt !== null) || a.rank - b.rank)
  const pageCount = Math.max(1, Math.ceil(nodes.length / NODES_PER_PAGE))
  const page = Math.min(requestedPage, pageCount - 1)
  const pageNodes = nodes.slice(page * NODES_PER_PAGE, (page + 1) * NODES_PER_PAGE)
  const nodeQuery = (selected: Node[]) => selected.map((n) => `node=${n.id}`).join('&')
  const openNodes = pageNodes.filter((n) => expanded.has(n.id))
  const [latestBody, seriesBody] = await Promise.all([
    pageNodes.length === 0 ? undefined : getJson($, `${base}/metrics/latest?${nodeQuery(pageNodes)}`),
    openNodes.length === 0
      ? undefined
      : getJson($, `${base}/metrics?since=${start}&step=${SPARK_STEP_MS}&agg=avg&name=cpu&name=gpu_util&${nodeQuery(openNodes)}`),
  ])

  const latest = new Map(
    items(latestBody, (item): [string, number] | undefined => {
      const node = str(at(item, 'node'))
      const name = str(at(item, 'name'))
      const value = num(at(item, 'value'))
      return node === undefined || name === undefined || value === undefined ? undefined : [metricKey(node, name), value]
    }),
  )

  const sparks = new Map<string, (number | null)[]>()
  for (const series of list(at(seriesBody, 'series'))) {
    const node = str(at(series, 'node'))
    const name = str(at(series, 'name'))
    if (node === undefined || name === undefined) continue
    const values = list(at(series, 'values'))
    const buckets = Array.from({ length: SPARK_BUCKETS }, (): number | null => null)
    list(at(series, 'at')).forEach((bucketAt, i) => {
      const ms = num(bucketAt)
      const value = num(values[i])
      if (ms === undefined || value === undefined) return
      const index = Math.floor((ms - start) / SPARK_STEP_MS)
      if (index >= 0 && index < SPARK_BUCKETS) buckets[index] = value
    })
    sparks.set(metricKey(node, name), buckets)
  }

  return { kind: 'detail', detail: { compute, nodes, page, pageCount, latest, sparks, tasks, log }, at: Date.now() }
}

// The reconciler's own writes also move the revision, so a refused If-Match is read again and sent again, under the same key.
async function deleteCompute($: EngineInterface, id: string): Promise<void> {
  const key = crypto.randomUUID()
  for (let attempt = 1; ; attempt++) {
    const revision = num(at(await getJson($, `/v1/computes/${id}`), 'revision'))
    if (revision === undefined) throw new Error(`unexpected compute payload for ${id}`)
    const response = await $.http.fetch(`${DAEMON}/v1/computes/${id}`, {
      method: 'DELETE',
      headers: { 'If-Match': `"${revision}"`, 'Idempotency-Key': key },
    })
    if (response.ok) return
    if (response.status !== 412 || attempt === DELETE_ATTEMPTS) throw refusal(response)
  }
}

export const register: Register = (on) => {
  let view: View = { kind: 'list' }
  let snapshot: Snapshot = { kind: 'loading' }
  let loading: View | undefined
  let timers: Timer[] = []
  let refresh: (() => void) | undefined
  let remove: ((id: string) => Promise<void>) | undefined
  let focusKey: string | undefined
  let notice: string | undefined
  let scrolling = false
  let removal: Removal | undefined
  const functionNames = new Map<string, string | null>()

  on('session.start', async ($, e, next) => {
    await $.command.register({ name: 'sky', description: 'Skyward computes panel' })
    return next(e)
  })

  on('command.run', { command: 'sky' }, async ($) => {
    if (timers.length > 0) {
      await $.ui.close({ id: PANE_ID })
      return { text: 'Skyward panel hidden' }
    }
    await $.ui.open({ id: PANE_ID, title: 'Skyward' })
    const tick = async () => {
      const target = view
      if (loading === target) return
      loading = target
      const result = await (target.kind === 'list' ? loadList($) : loadDetail($, target, snapshot.kind === 'detail' ? snapshot.detail.log : undefined, functionNames))
        .catch((error: unknown): Snapshot => ({ kind: 'down', reason: error instanceof Error ? error.message : String(error) }))
        .finally(() => {
          if (loading === target) loading = undefined
        })
      if (view !== target) return
      snapshot = result
      $.ui.invalidate('ui.render')
    }
    refresh = () => void tick()
    remove = (id) => deleteCompute($, id)
    timers = [
      $.clock.every(REFRESH_MS, refresh),
      $.clock.every(MARQUEE_MS, () => {
        if (scrolling) $.ui.invalidate('ui.render')
      }),
    ]
    refresh()
    return { text: 'Skyward panel shown' }
  })

  on('ui.close', { id: PANE_ID }, async ($, e, next) => {
    const result = await next(e)
    timers.forEach((timer) => timer.cancel())
    timers = []
    refresh = undefined
    remove = undefined
    view = { kind: 'list' }
    snapshot = { kind: 'loading' }
    removal = undefined
    return result
  })

  on('ui.render', { component: 'Pane', requestId: PANE_ID }, ($, e) => {
    const { Box, Button, Text } = $.ui.resolve(e)
    const now = Date.now()

    const show = (next: View, focus: string) => {
      view = next
      snapshot = { kind: 'loading' }
      focusKey = focus
      notice = undefined
      removal = undefined
      refresh?.()
      $.ui.invalidate('ui.render')
    }

    // Keeps the current snapshot on screen until the page arrives, and both buttons drawn so the focus ring stays put.
    const turnPage = (page: number, pageCount: number) => {
      if (view.kind !== 'detail' || page < 0 || page >= pageCount) return
      view = { ...view, page }
      refresh?.()
    }

    const filterLog = (log: LogFilter, page?: number) => {
      if (view.kind !== 'detail') return
      if (page !== undefined && page !== view.page) focusKey = 'log-node'
      view = { ...view, log, page: page ?? view.page }
      notice = undefined
      refresh?.()
      $.ui.invalidate('ui.render')
    }

    const updateTasks = (tasks: TaskQuery) => {
      if (view.kind !== 'detail') return
      view = { ...view, tasks }
      refresh?.()
      $.ui.invalidate('ui.render')
    }

    const toggleLogTask = (taskId: string) => {
      if (view.kind !== 'detail') return
      filterLog({ ...view.log, task: view.log.task === taskId ? null : taskId, cursors: [] })
    }

    const toggleLogNode = (nodeId: string) => {
      if (view.kind !== 'detail') return
      filterLog({ ...view.log, node: view.log.node === nodeId ? null : nodeId, cursors: [] })
    }

    const toggleNode = (nodeId: string) => {
      if (view.kind !== 'detail') return
      const expanded = new Set(view.expanded)
      if (!expanded.delete(nodeId)) expanded.add(nodeId)
      view = { ...view, expanded }
      refresh?.()
      $.ui.invalidate('ui.render')
    }

    const setRemoval = (next: Removal | undefined, focus: string) => {
      removal = next
      focusKey = focus
      $.ui.invalidate('ui.render')
    }

    const confirmRemoval = (id: string) => {
      setRemoval({ id, step: 'sending' }, 'back')
      remove?.(id).then(
        () => refresh?.(),
        (error: unknown) => {
          if (removal?.id !== id) return
          removal = { id, step: 'failed', reason: error instanceof Error ? error.message : String(error) }
          $.ui.invalidate('ui.render')
        },
      )
    }

    // The first press only asks, with the ring on cancel, so a second Enter does not delete.
    const removeControls = (c: Compute): RenderElement | null => {
      if (!deletable(c)) return null
      const pending = removal?.id === c.id ? removal : undefined
      if (pending?.step === 'confirm') {
        return (
          <Box flexDirection="row" flexWrap="wrap" columnGap={1}>
            <Text color="red">delete this compute?</Text>
            <Button key="delete-confirm" label="yes, delete" onPress={() => confirmRemoval(c.id)} />
            <Button key="delete-cancel" label="cancel" onPress={() => setRemoval(undefined, 'delete')} />
          </Box>
        )
      }
      if (pending?.step === 'sending') return <Text dimColor>deleting…</Text>
      return <Button key="delete" label="delete" onPress={() => setRemoval({ id: c.id, step: 'confirm' }, 'delete-cancel')} />
    }

    scrolling = false
    const columns = fitColumns(e.props.bodyColumns)
    const tableWidth = columns.reduce((width, column) => width + TABLE_GAP + column.width, NAME_WIDTH)

    const cell = (column: Column, text: string, props: { color?: string; dim?: boolean; bold?: boolean }): RenderElement => (
      <Box width={column.width} flexShrink={0}>
        <Text wrap="truncate" color={props.color} dimColor={props.dim} bold={props.bold}>
          {column.right ? text.padStart(column.width) : text}
        </Text>
      </Box>
    )

    const tableRow = (name: RenderElement, cells: RenderElement[]): RenderElement => (
      <Box flexDirection="row" gap={TABLE_GAP}>
        <Box width={NAME_WIDTH} flexShrink={0}>{name}</Box>
        {cells}
      </Box>
    )

    const computeRow = (c: Compute): RenderElement => {
      const name = c.name ?? c.id
      if (name.length > NAME_WIDTH) scrolling = true
      return (
        <Box flexDirection="column">
          {tableRow(
            <Button key={c.id} plain label={marquee(name, NAME_WIDTH, now)} dimColor={c.ended} onPress={() => show({ kind: 'detail', id: c.id, page: 0, expanded: new Set(), tasks: { order: 'state', fn: null, cursors: [] }, log: { node: null, task: null, term: '', cursors: [] } }, 'back')} />,
            columns.map((column) => {
              const { text, color } = column.cell(c)
              return cell(column, text, { color: c.ended ? undefined : color, dim: c.ended })
            }),
          )}
          {c.lastError !== null && !c.ended ? <Text color="red" wrap="truncate">{`  ${c.lastError}`}</Text> : null}
        </Box>
      )
    }

    const rule = (): RenderElement => <Text dimColor>{'─'.repeat(tableWidth)}</Text>

    const computeTable = (live: Compute[], recent: Compute[]): RenderElement => (
      <Box flexDirection="column">
        {tableRow(<Text dimColor>NAME</Text>, columns.map((column) => cell(column, column.title, { dim: true })))}
        {rule()}
        {live.length === 0 ? <Text dimColor>no live computes</Text> : live.map(computeRow)}
        {live.length === 0
          ? null
          : [
              rule(),
              tableRow(
                <Text dimColor>total</Text>,
                columns.map((column) => cell(column, column.total?.(live) ?? '', { bold: true })),
              ),
            ]}
        {recent.length === 0 ? null : [<Text> </Text>, tableRow(<Text dimColor>recent</Text>, []), ...recent.map(computeRow)]}
      </Box>
    )

    const computeHeader = (c: Compute): RenderElement => (
      <Box flexDirection="column">
        <Box flexDirection="row" gap={2}>
          <Text bold wrap="truncate">{c.name ?? c.id}</Text>
          <Text color={STATE_COLORS.get(c.state)}>{`${STATE_GLYPHS.get(c.state) ?? '●'} ${c.state}`}</Text>
        </Box>
        <Text dimColor wrap="truncate">
          {[
            `${c.nodesReady}/${c.nodesTotal} nodes`,
            c.accelerator === null ? c.provider : `${c.provider} ${gpu(c)}`,
            c.ended ? `cost $${money(c.cost)}` : `$${money(hourly(c))}/h`,
            age(now - c.createdAt),
          ].join(' · ')}
        </Text>
        {c.lastError !== null ? <Text color="red" wrap="truncate">{c.lastError}</Text> : null}
      </Box>
    )

    const rankBadge = (rank: number | undefined, muted = false): RenderElement =>
      rank === undefined || muted ? (
        <Text dimColor>{rank === undefined ? ' #? ' : ` #${rank} `}</Text>
      ) : (
        <Text bold color="#ffffff" backgroundColor={RANK_COLORS[rank % RANK_COLORS.length]}>{` #${rank} `}</Text>
      )

    const nodeBlock = (n: Node, d: Detail, sparkStart: number, open: boolean, logging: boolean): RenderElement => {
      const terminated = n.terminatedAt !== null
      const metric = (name: string) => d.latest.get(metricKey(n.id, name))
      const both = (a: number | undefined, b: number | undefined, f: (a: number, b: number) => string) =>
        a === undefined || b === undefined ? null : f(a, b)
      const one = (a: number | undefined, f: (a: number) => string) => (a === undefined ? null : f(a))

      const stats = (parts: [string, string | null][], indent = 4): RenderElement => (
        <Box flexDirection="row" flexWrap="wrap" columnGap={2} paddingLeft={indent}>
          {parts.map(([label, value]) =>
            value === null ? null : (
              <Box flexShrink={0}>
                <Text><Text dimColor>{`${label} `}</Text>{value}</Text>
              </Box>
            ),
          )}
        </Box>
      )

      const trend = (label: string, name: string, parts: [string, string | null][]): RenderElement | null =>
        metric(name) === undefined ? null : (
          <Box flexDirection="row" gap={2}>
            <Box flexShrink={0}>
              <Text>{`    ${label} ${spark((d.sparks.get(metricKey(n.id, name)) ?? NO_SAMPLES).slice(sparkStart))}`}</Text>
            </Box>
            <Box flexShrink={1}>{stats(parts, 0)}</Box>
          </Box>
        )

      const life =
        n.launchedAt === null ? null : n.terminatedAt === null ? `up ${age(now - n.launchedAt)}` : `ran ${age(n.terminatedAt - n.launchedAt)}`
      const tail = [life, n.price === null ? null : `@ ${price(n.price)}`, n.market === null ? null : `(${n.market.replace('_', ' ')})`]
        .filter((part) => part !== null)
        .join(' ')

      return (
        <Box flexDirection="column">
          <Box flexDirection="row" gap={1}>
            <Button key={`node:${n.id}`} plain label={open ? '▾' : '▸'} onPress={() => toggleNode(n.id)} />
            <Box flexShrink={0}>{rankBadge(n.rank, terminated)}</Box>
            <Text>
              <Text dimColor>is </Text>
              <Text color={terminated ? undefined : STATE_COLORS.get(n.state)} dimColor={terminated}>{n.state}</Text>
              <Text dimColor>{tail === '' ? '' : ` ${tail}`}</Text>
            </Text>
            <Box flexShrink={0} paddingLeft={1}>
              <Button key={`logs:${n.id}`} plain label="logs" dimColor={!logging} onPress={() => toggleLogNode(n.id)} />
            </Box>
          </Box>
          {n.lastError !== null ? <Text color="red">{`    ${n.lastError}`}</Text> : null}
          {terminated
            ? null
            : stats([
                ['cpu', one(metric('cpu'), (v) => `${Math.round(v)}%`)],
                ['gpu', one(metric('gpu_util'), (v) => `${Math.round(v)}%`)],
                ['memory', both(metric('mem_used_mb'), metric('mem_total_mb'), (used, total) => `${gib(used)}/${gib(total)} GB`)],
                ['disk', one(metric('disk_used_pct'), (v) => `${Math.round(v)}%`)],
              ])}
          {open && !terminated
            ? [
                trend('cpu', 'cpu', [['net', both(metric('net_rx_kbps'), metric('net_tx_kbps'), (rx, tx) => `↓${rate(rx)} ↑${rate(tx)}`)]]),
                trend('gpu', 'gpu_util', [
                  ['vram', both(metric('gpu_mem_mb'), metric('gpu_mem_total_mb'), (used, total) => `${gib(used)}/${gib(total)} GB`)],
                  ['temp', one(metric('gpu_temp_c'), (v) => `${Math.round(v)}°C`)],
                  ['power', one(metric('gpu_power_w'), (v) => `${Math.round(v)} W`)],
                ]),
              ]
            : null}
          {open ? (
            <Box paddingLeft={4}>
              <Text dimColor>{[n.address, n.id, n.machine === null ? null : `machine ${n.machine}`].filter((part) => part !== null).join(' · ')}</Text>
            </Box>
          ) : null}
        </Box>
      )
    }

    const body = (): RenderElement => {
      switch (snapshot.kind) {
        case 'loading':
          return <Text dimColor>loading…</Text>
        case 'down':
          return (
            <Box flexDirection="column">
              <Text color="red" wrap="truncate">{`daemon request failed at ${DAEMON}: ${snapshot.reason}`}</Text>
              <Text dimColor>sky server start</Text>
            </Box>
          )
        case 'list':
          return (
            <Box flexDirection="column" gap={1}>
              <Text dimColor>{`${DAEMON} · updated ${clock(snapshot.at)}`}</Text>
              {computeTable(snapshot.live, snapshot.recent)}
            </Box>
          )
        case 'detail': {
          const d = snapshot.detail
          const ranks = new Map(d.nodes.map((n) => [n.id, n.rank]))
          const first = d.page * NODES_PER_PAGE
          const pageNodes = d.nodes.slice(first, first + NODES_PER_PAGE)
          // Columns before any node on the page had a sample are blank for all of them, so they are dropped.
          const sparkStart = Math.min(SPARK_BUCKETS, ...[...d.sparks.values()].map((buckets) => buckets.findIndex((value) => value !== null)).filter((i) => i >= 0))
          return (
            <Box flexDirection="column" gap={1}>
              <Box flexDirection="column">
                {computeHeader(d.compute)}
                <Text dimColor>{`${d.compute.id} · updated ${clock(snapshot.at)}`}</Text>
              </Box>
              <Box flexDirection="column">
                <Box flexDirection="row" gap={2}>
                  <Text bold>{`Nodes (${d.nodes.length})`}</Text>
                  {d.pageCount > 1 ? (
                    <Box flexDirection="row" gap={1}>
                      <Button key="nodes-prev" plain label="‹ prev" dimColor={d.page === 0} onPress={() => turnPage(d.page - 1, d.pageCount)} />
                      <Text dimColor>{`${first + 1}–${first + pageNodes.length} of ${d.nodes.length}`}</Text>
                      <Button key="nodes-next" plain label="next ›" dimColor={d.page === d.pageCount - 1} onPress={() => turnPage(d.page + 1, d.pageCount)} />
                    </Box>
                  ) : null}
                </Box>
                {d.nodes.length === 0 ? <Text dimColor>no nodes</Text> : pageNodes.map((n) => nodeBlock(n, d, sparkStart, view.kind === 'detail' && view.expanded.has(n.id), view.kind === 'detail' && view.log.node === n.id))}
              </Box>
              {view.kind === 'detail' ? tasksSection(d, view, ranks) : null}
              {view.kind === 'detail' ? logSection(d, view, ranks) : null}
            </Box>
          )
        }
      }
    }

    const taskRow = (t: Task, query: TaskQuery, ranks: Map<string, number>, logging: boolean): RenderElement => {
      const state = t.state.replace('_', ' ')
      const took = t.startedAt === null ? null : age((t.finishedAt ?? now) - t.startedAt)
      const status =
        t.state === 'queued'
          ? [`queued ${age(now - t.submittedAt)}`]
          : t.finishedAt === null
            ? [took === null ? state : `${state} ${took}`]
            : [took === null ? state : `${state} ${t.state === 'succeeded' ? 'in' : 'after'} ${took}`, `${age(now - t.finishedAt)} ago`]
      const name = t.name
      return (
        <Box flexDirection="column">
          <Box flexDirection="row" gap={1}>
            <Box width={1} flexShrink={0}>
              <Text color={TASK_COLORS.get(t.state)} dimColor={!TASK_COLORS.has(t.state)}>{TASK_GLYPHS.get(t.state) ?? '·'}</Text>
            </Box>
            <Box width={RANK_WIDTH} flexShrink={0}>{t.node === null ? null : rankBadge(ranks.get(t.node))}</Box>
            <Box flexShrink={0}>
              {name === null ? (
                <Text dimColor>{t.sha.slice(0, 12)}</Text>
              ) : (
                // The filtered page may not hold this row, so the ring is handed to the sort button, which every page draws.
                <Button key={`task-fn:${t.id}`} plain label={name} onPress={() => {
                    focusKey = 'tasks-sort'
                    updateTasks({ ...query, fn: query.fn === name ? null : name, cursors: [] })
                  }} />
              )}
            </Box>
            <Text dimColor>{[...status, t.attempts > 1 ? `${t.attempts} attempts` : null].filter((part) => part !== null).join(' · ')}</Text>
            {t.state === 'queued' ? null : (
              <Box flexShrink={0} paddingLeft={1}>
                <Button key={`task-logs:${t.id}`} plain label="logs" dimColor={!logging} onPress={() => toggleLogTask(t.id)} />
              </Box>
            )}
          </Box>
          {t.error === null ? null : (
            <Box paddingLeft={RANK_WIDTH + 3}>
              <Text color="red" wrap="wrap">{t.error}</Text>
            </Box>
          )}
        </Box>
      )
    }

    const tasksSection = (d: Detail, target: DetailView, ranks: Map<string, number>): RenderElement => {
      const query = target.tasks
      const page = d.tasks
      const first = query.cursors.length * TASKS_PER_PAGE
      const count = (state: string) => page.counts.get(state) ?? 0
      const other = (page.total ?? 0) - COUNTED_TASK_STATES.reduce((sum, state) => sum + count(state), 0)
      const summary = [...COUNTED_TASK_STATES.map((state) => `${count(state)} ${state}`), other > 0 ? `${other} other` : null]
        .filter((part) => part !== null)
        .join(' · ')
      const next = page.next
      const order = TASK_ORDERS[(TASK_ORDERS.indexOf(query.order) + 1) % TASK_ORDERS.length]
      const functionInput = (): RenderElement | null => {
        if (e.surface === 'mobile') return null
        const { Input } = $.ui.resolve(e)
        return (
          <Input key="tasks-function" label="function" placeholder={query.fn ?? 'all'} submitLabel="filter" onSubmit={(value) => updateTasks({ ...query, fn: value.trim() === '' ? null : value.trim(), cursors: [] })} />
        )
      }
      return (
        <Box flexDirection="column">
          <Box flexDirection="row" flexWrap="wrap" columnGap={2}>
            <Text bold>{`Tasks (${page.total ?? '?'})`}</Text>
            <Text dimColor>{summary}</Text>
          </Box>
          <Box flexDirection="row" flexWrap="wrap" columnGap={2}>
            <Button key="tasks-sort" plain label={`sort: ${query.order} ›`} onPress={() => updateTasks({ ...query, order, cursors: [] })} />
            {functionInput()}
            <Box flexDirection="row" gap={1}>
              <Button key="tasks-prev" plain label="‹ prev" dimColor={query.cursors.length === 0} onPress={() => updateTasks({ ...query, cursors: query.cursors.slice(0, -1) })} />
              <Text dimColor>{`${page.items.length === 0 ? first : `${first + 1}–${first + page.items.length}`} of ${page.total ?? '?'}`}</Text>
              <Button key="tasks-next" plain label="next ›" dimColor={next === null} onPress={() => {
                  if (next !== null) updateTasks({ ...query, cursors: [...query.cursors, next] })
                }} />
            </Box>
          </Box>
          {page.items.length === 0 ? <Text dimColor>{query.fn === null ? 'no tasks' : `no ${query.fn} tasks`}</Text> : null}
          {page.items.map((t) => taskRow(t, query, ranks, target.log.task === t.id))}
        </Box>
      )
    }

    const logSection = (d: Detail, target: DetailView, ranks: Map<string, number>): RenderElement => {
      const filter = target.log
      const newest = filter.cursors.length === 0
      const next = d.log.next
      const pickNode = (value: string) => {
        const text = value.trim().replace(/^#/, '')
        if (text === '') return filterLog({ ...filter, node: null, cursors: [] })
        // A replaced node leaves its rank to the next one, so a rank can name several machines: the live one wins, then the latest.
        const matches = d.nodes.filter((n) => n.rank === Number(text))
        const node = matches.find((n) => n.terminatedAt === null) ?? matches.sort((a, b) => (b.launchedAt ?? 0) - (a.launchedAt ?? 0))[0]
        if (node === undefined) {
          notice = `no node #${text}`
          $.ui.invalidate('ui.render')
          return
        }
        filterLog({ ...filter, node: node.id, cursors: [] }, Math.floor(d.nodes.indexOf(node) / NODES_PER_PAGE))
      }
      // Mobile has no Input; there each node's logs button is the node filter. A field empties on Enter and is only
      // refilled by a value that differs from the last one drawn, so the filter in force is shown as the placeholder.
      const inputs = (): RenderElement[] => {
        if (e.surface === 'mobile') return []
        const { Input } = $.ui.resolve(e)
        return [
          <Input key="log-node" label="node" placeholder={filter.node === null ? 'all' : `#${ranks.get(filter.node) ?? '?'}`} submitLabel="go" onSubmit={pickNode} />,
          <Input key="log-term" label="filter" placeholder={filter.term === '' ? 'term' : filter.term} submitLabel="search" onSubmit={(value) => filterLog({ ...filter, term: value.trim(), cursors: [] })} />,
        ]
      }
      return (
        <Box flexDirection="column">
          <Box flexDirection="row" flexWrap="wrap" columnGap={2}>
            <Text bold>Log</Text>
            {inputs()}
            {filter.task === null ? null : (
              <Button key="log-task" plain label={`task ${filter.task} ✕`} onPress={() => {
                  focusKey = 'log-older'
                  filterLog({ ...filter, task: null, cursors: [] })
                }} />
            )}
            {notice === undefined ? null : <Text color="red">{notice}</Text>}
          </Box>
          <Box flexDirection="row" gap={1}>
            <Button key="log-older" plain label="‹ older" dimColor={next === null} onPress={() => {
                if (next !== null) filterLog({ ...filter, cursors: [...filter.cursors, next] })
              }} />
            <Text dimColor>{newest ? 'live' : `page ${filter.cursors.length + 1}`}</Text>
            <Button key="log-newer" plain label="newer ›" dimColor={newest} onPress={() => filterLog({ ...filter, cursors: filter.cursors.slice(0, -1) })} />
            <Button key="log-latest" plain label="latest »" dimColor={newest} onPress={() => filterLog({ ...filter, cursors: [] })} />
          </Box>
          {d.log.lines.length === 0 ? <Text dimColor>{filter.node === null && filter.task === null && filter.term === '' ? 'no events' : 'no matching events'}</Text> : null}
          {d.log.lines.map((l) => (
            <Box flexDirection="row" gap={1}>
              <Box width={8} flexShrink={0}><Text dimColor>{clock(l.at)}</Text></Box>
              <Box width={RANK_WIDTH} flexShrink={0}>
                {l.node === null ? null : rankBadge(ranks.get(l.node))}
              </Box>
              <Box flexShrink={1}>
                <Text wrap="wrap" color={l.error ? 'red' : undefined} dimColor={!l.error && l.type !== 'node.console'}>{l.text}</Text>
              </Box>
            </Box>
          ))}
        </Box>
      )
    }

    const current = view
    const shown = snapshot.kind === 'detail' && current.kind === 'detail' && snapshot.detail.compute.id === current.id ? snapshot.detail.compute : undefined
    const tree =
      current.kind === 'list' ? (
        body()
      ) : (
        <Box flexDirection="column" gap={1}>
          <Box flexDirection="column">
            <Box flexDirection="row" flexWrap="wrap" columnGap={2}>
              <Button key="back" label="← computes" onPress={() => show({ kind: 'list' }, current.id)} />
              {shown === undefined ? null : removeControls(shown)}
            </Box>
            {shown !== undefined && deletable(shown) && removal?.step === 'failed' && removal.id === shown.id ? (
              <Text color="red">{`delete failed: ${removal.reason}`}</Text>
            ) : null}
          </Box>
          {body()}
        </Box>
      )

    // A pressed button missing from the new tree leaves the ring on the pane's close mark, and a page of nodes swapped
    // above the focused node field returns the keys to the prompt; either way the ring is handed back once that tree is drawn.
    const key = focusKey
    const drawn = current.kind === 'list' ? snapshot.kind === 'list' : snapshot.kind !== 'detail' || snapshot.detail.page === current.page
    if (key !== undefined && drawn) {
      focusKey = undefined
      if (e.props.isFocused) $.clock.after(0, () => void $.ui.focus({ requestId: PANE_ID, key }))
    }
    return tree
  })
}
