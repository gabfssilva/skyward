import type { components, paths } from './schema'

export type Schemas = components['schemas']
export type Compute = Schemas['ComputeResource']
export type ComputeCreate = Schemas['CreateComputeResource']
export type ComputeSpec = Schemas['ComputeSpec']
export type ComputeSpecPatch = Schemas['UpdateComputeResource']
export type ComputeStatus = Schemas['ComputeStatus']
export type NodeBounds = Schemas['NodeBounds']
export type Spec = Schemas['Spec']
export type Image = Schemas['Image']
export type Worker = Schemas['Worker']
export type Options = Schemas['Options']
export type Result = Schemas['CommandResultResource']
export type PluginRef = Schemas['PluginRef']
export type Lease = Schemas['LeaseResource']
export type Ending = Schemas['Ending']
export type Node = Schemas['NodeResource']
export type Ssh = Schemas['Ssh']
export type Progress = Schemas['Progress']
export type Task = Schemas['TaskResource']
export type TaskCounts = Schemas['TaskCounts']
export type TaskCreate = Schemas['CreateTaskResource']
export type FunctionSummary = Schemas['FunctionSummary']
export type Execution = Schemas['ExecutionResource']
export type ExecutionCreate = Schemas['CreateExecutionResource']
export type Provider = Schemas['ProviderResource']
export type ProviderCreate = Schemas['CreateProviderResource']
export type ProviderKind = Schemas['ProviderKindResource']
export type Offer = Schemas['OfferResource']
export type Accelerator = Schemas['AcceleratorResource']
export type FunctionRef = Schemas['FunctionResource']
export type FunctionSource = Schemas['WriteFunctionResource']
export type Call = Schemas['Call']
export type Generation = Schemas['GenerationResource']
export type Liveness = Schemas['LivenessResource']
export type LogEntry = Schemas['LogEntryResource']
export type WireError = Schemas['Error']
export type ErrorCode = WireError['code']

/**
 * A slice of a listing, the cursor that continues it, and how many rows there are.
 *
 * ``next_cursor`` is null on the last page; cursors are opaque, never offsets.
 * ``total`` is how many the filters match, which is not how many the page carries —
 * it is what lets a list say fifty of nine thousand instead of fifty of fifty.
 */
export type Page<T> = { items: T[]; next_cursor?: string | null; total?: number | null }

/** How the daemon orders a page of the catalog. */
export type OfferSort = NonNullable<NonNullable<paths['/v1/offers']['get']['parameters']['query']>['sort']>

/** How the daemon orders a page of tasks. */
export type TaskOrder = NonNullable<NonNullable<paths['/v1/tasks']['get']['parameters']['query']>['order']>

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
    readonly code?: ErrorCode,
    readonly retryable = false,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

export const BASE = '/v1'

type Query = Record<string, string | number | boolean | null | undefined | readonly string[]>

const qs = (query?: Query): string => {
  if (!query) return ''
  const params = new URLSearchParams()
  for (const [key, value] of Object.entries(query)) {
    if (value === undefined || value === null) continue
    if (Array.isArray(value)) for (const v of value) params.append(key, v)
    else params.append(key, String(value))
  }
  const s = params.toString()
  return s ? `?${s}` : ''
}

const failure = async (res: Response): Promise<ApiError> => {
  const body = await res.text()
  try {
    const parsed = JSON.parse(body) as Partial<WireError> & { detail?: string }
    const message = parsed.message ?? parsed.detail ?? body
    return new ApiError(res.status, message, parsed.code, parsed.retryable ?? false)
  } catch {
    return new ApiError(res.status, body || res.statusText)
  }
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { 'content-type': 'application/json', ...(init?.headers ?? {}) },
  })
  if (!res.ok) throw await failure(res)
  if (res.status === 204) return undefined as T
  const text = await res.text()
  return (text ? JSON.parse(text) : undefined) as T
}

const body = (value: unknown): { body: string } => ({ body: JSON.stringify(value) })

/**
 * A fresh ``Idempotency-Key``, the one header every create and delete must carry.
 *
 * Hex like the ``uuid4().hex`` the CLI sends, drawn from ``getRandomValues`` because
 * ``randomUUID`` needs a secure context and a daemon bound to another host is opened over plain http.
 */
const once = (): Record<string, string> => ({
  'Idempotency-Key': Array.from(crypto.getRandomValues(new Uint8Array(16)), (b) => b.toString(16).padStart(2, '0')).join(''),
})

/** How many times a compute write re-reads the revision before a conflict is handed back, as ``sky`` does. */
const WRITE_ATTEMPTS = 5

/**
 * A write against a compute, guarded by ``If-Match`` on the revision it was just read at.
 *
 * The revision also moves on bookkeeping nobody asked for — the reconciler writes what it observed on every
 * tick — so a ``revision_conflict`` re-reads and re-sends the same write, key included, rather than failing a
 * change nothing was racing.
 */
async function conditional<T>(id: string, init: RequestInit & { headers?: Record<string, string> }, attempts = WRITE_ATTEMPTS): Promise<T> {
  const current = await request<Compute>(`/computes/${id}`)
  try {
    return await request<T>(`/computes/${id}`, { ...init, headers: { ...init.headers, 'If-Match': `"${current.revision}"` } })
  } catch (error) {
    if (error instanceof ApiError && error.code === 'revision_conflict' && attempts > 1) return conditional(id, init, attempts - 1)
    throw error
  }
}

export type ComputeQuery = { cursor?: string; limit?: number; state?: string; owned?: boolean; live?: boolean; cause?: Ending['cause']; include?: string }
/** Blocks a compute carries beyond the default, comma-separated: what ``?include=`` takes. */
export type NodeQuery = { include?: string }
/** ``latest`` is one row per function, its newest upload; ``lineage`` is every upload of one function. */
export type FunctionQuery = { cursor?: string; limit?: number; latest?: boolean; lineage?: string }
export type TaskQuery = { cursor?: string; limit?: number; compute?: string; state?: string; correlation_id?: string; order?: TaskOrder }
export type LogQuery = { cursor?: string; limit?: number; compute?: string; task?: string; node?: string; types?: readonly string[]; contains?: readonly string[] }
export type OfferQuery = {
  provider?: string
  kind?: string
  accelerator?: string
  min_count?: number
  min_vram?: number
  max_price?: number
  refresh?: boolean
  spot?: boolean
  sort?: OfferSort
  limit?: number
}

export const api = {
  health: (): Promise<Liveness> => request<Liveness>('/health/live'),

  computes: (query?: ComputeQuery): Promise<Page<Compute>> => request<Page<Compute>>(`/computes${qs(query)}`),
  compute: (id: string, include?: string): Promise<Compute> => request<Compute>(`/computes/${id}${qs({ include })}`),
  createCompute: (payload: ComputeCreate): Promise<Compute> => request<Compute>('/computes', { method: 'POST', ...body(payload), headers: once() }),
  patchCompute: (id: string, patch: ComputeSpecPatch): Promise<Compute> => conditional<Compute>(id, { method: 'PATCH', ...body(patch) }),
  scale: (id: string, nodes: NodeBounds): Promise<Compute> => conditional<Compute>(id, { method: 'PATCH', ...body({ nodes }) }),
  deleteCompute: (id: string): Promise<void> => conditional<void>(id, { method: 'DELETE', headers: once() }),
  generations: (id: string): Promise<Page<Generation>> => request<Page<Generation>>(`/computes/${id}/generations`),

  nodes: (computeId: string, query?: NodeQuery): Promise<Page<Node>> => request<Page<Node>>(`/computes/${computeId}/nodes${qs(query)}`),
  node: (computeId: string, nodeId: string): Promise<Node> => request<Node>(`/computes/${computeId}/nodes/${nodeId}`),
  drainNode: (computeId: string, nodeId: string): Promise<void> => request<void>(`/computes/${computeId}/nodes/${nodeId}`, { method: 'DELETE', headers: once() }),

  exec: (computeId: string, command: string, node?: number): Promise<Record<string, Result>> =>
    request<Record<string, Result>>(`/computes/${computeId}/exec${qs({ command, node })}`, { method: 'POST' }),

  tasks: (query?: TaskQuery): Promise<Page<Task>> => request<Page<Task>>(`/tasks${qs(query)}`),
  submitTask: (payload: TaskCreate): Promise<Task> => request<Task>('/tasks', { method: 'POST', ...body(payload), headers: once() }),
  task: (id: string): Promise<Task> => request<Task>(`/tasks/${id}`),
  cancelTask: (id: string): Promise<void> => request<void>(`/tasks/${id}`, { method: 'DELETE', headers: once() }),
  executions: (taskId: string): Promise<Page<Execution>> => request<Page<Execution>>(`/tasks/${taskId}/executions`),
  retry: (taskId: string, payload: ExecutionCreate = { acknowledge_duplication: false }): Promise<Execution> =>
    request<Execution>(`/tasks/${taskId}/executions`, { method: 'POST', ...body(payload), headers: once() }),

  functions: (query?: FunctionQuery): Promise<Page<FunctionRef>> => request<Page<FunctionRef>>(`/functions${qs(query)}`),
  function: (sha256: string): Promise<FunctionRef> => request<FunctionRef>(`/functions/${sha256}`),
  writeFunction: (payload: FunctionSource): Promise<FunctionRef> => request<FunctionRef>('/functions', { method: 'POST', ...body(payload) }),

  log: (query?: LogQuery): Promise<Page<LogEntry>> => request<Page<LogEntry>>(`/events/log${qs(query)}`),

  providers: (): Promise<Page<Provider>> => request<Page<Provider>>('/providers'),
  provider: (id: string): Promise<Provider> => request<Provider>(`/providers/${id}`),
  createProvider: (payload: ProviderCreate): Promise<Provider> => request<Provider>('/providers', { method: 'POST', ...body(payload) }),
  updateProvider: (id: string, payload: ProviderCreate): Promise<Provider> => request<Provider>(`/providers/${id}`, { method: 'PUT', ...body(payload) }),
  deleteProvider: (id: string): Promise<void> => request<void>(`/providers/${id}`, { method: 'DELETE' }),
  providerKinds: (): Promise<ProviderKind[]> => request<ProviderKind[]>('/provider-kinds'),

  offers: (query?: OfferQuery): Promise<Page<Offer>> => request<Page<Offer>>(`/offers${qs(query)}`),
  accelerators: (): Promise<Accelerator[]> => request<Accelerator[]>('/accelerators'),
}
