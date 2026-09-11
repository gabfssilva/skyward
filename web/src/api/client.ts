import type { components } from './schema'

export type Schemas = components['schemas']
export type Compute = Schemas['Compute']
export type ComputeCreate = Schemas['ComputeCreate']
export type ComputeSpec = Schemas['ComputeSpec']
export type ComputeSpecPatch = Schemas['ComputeSpecPatch']
export type ComputeStatus = Schemas['ComputeStatus']
export type NodeBounds = Schemas['NodeBounds']
export type Spec = Schemas['Spec']
export type Image = Schemas['Image']
export type Worker = Schemas['Worker']
export type Options = Schemas['Options']
export type Result = Schemas['Result']
export type PluginRef = Schemas['PluginRef']
export type Lease = Schemas['Lease']
export type Node = Schemas['Node']
export type Task = Schemas['Task']
export type TaskCreate = Schemas['TaskCreate']
export type Execution = Schemas['Execution']
export type ExecutionCreate = Schemas['ExecutionCreate']
export type Provider = Schemas['Provider']
export type ProviderCreate = Schemas['ProviderCreate']
export type ProviderKind = Schemas['ProviderKind']
export type Offer = Schemas['Offer']
export type FunctionRef = Schemas['Function']
export type Generation = Schemas['Generation']
export type Liveness = Schemas['Liveness']
export type WireError = Schemas['Error']
export type ErrorCode = WireError['code']

/**
 * A slice of a listing, and the cursor that continues it.
 *
 * ``next_cursor`` is null on the last page; cursors are opaque, never offsets.
 */
export type Page<T> = { items: T[]; next_cursor?: string | null }

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

const body = (value: unknown): RequestInit => ({ body: JSON.stringify(value) })

export type ComputeQuery = { cursor?: string; limit?: number; state?: string; owned?: boolean; live?: boolean }
export type NodeQuery = { include_terminal?: boolean; generation?: number }
export type TaskQuery = { cursor?: string; limit?: number; compute?: string; state?: string; correlation_id?: string }
export type OfferQuery = {
  provider?: string
  kind?: string
  accelerator?: string
  min_count?: number
  min_vram?: number
  max_price?: number
  refresh?: boolean
}

export const api = {
  health: (): Promise<Liveness> => request<Liveness>('/health/live'),

  computes: (query?: ComputeQuery): Promise<Page<Compute>> => request<Page<Compute>>(`/computes${qs(query)}`),
  compute: (id: string): Promise<Compute> => request<Compute>(`/computes/${id}`),
  createCompute: (payload: ComputeCreate): Promise<Compute> => request<Compute>('/computes', { method: 'POST', ...body(payload) }),
  patchCompute: (id: string, patch: ComputeSpecPatch): Promise<Compute> => request<Compute>(`/computes/${id}`, { method: 'PATCH', ...body(patch) }),
  scale: (id: string, nodes: NodeBounds): Promise<Compute> => request<Compute>(`/computes/${id}`, { method: 'PATCH', ...body({ nodes }) }),
  deleteCompute: (id: string): Promise<void> => request<void>(`/computes/${id}`, { method: 'DELETE' }),
  generations: (id: string): Promise<Page<Generation>> => request<Page<Generation>>(`/computes/${id}/generations`),

  nodes: (computeId: string, query?: NodeQuery): Promise<Page<Node>> => request<Page<Node>>(`/computes/${computeId}/nodes${qs(query)}`),
  node: (computeId: string, nodeId: string): Promise<Node> => request<Node>(`/computes/${computeId}/nodes/${nodeId}`),
  drainNode: (computeId: string, nodeId: string): Promise<void> => request<void>(`/computes/${computeId}/nodes/${nodeId}`, { method: 'DELETE' }),

  exec: (computeId: string, command: string, node?: number): Promise<Schemas['Result']> =>
    request<Schemas['Result']>(`/computes/${computeId}/exec${qs({ command, node })}`, { method: 'POST' }),

  tasks: (query?: TaskQuery): Promise<Page<Task>> => request<Page<Task>>(`/tasks${qs(query)}`),
  task: (id: string): Promise<Task> => request<Task>(`/tasks/${id}`),
  cancelTask: (id: string): Promise<void> => request<void>(`/tasks/${id}`, { method: 'DELETE' }),
  executions: (taskId: string): Promise<Page<Execution>> => request<Page<Execution>>(`/tasks/${taskId}/executions`),
  retry: (taskId: string, payload: ExecutionCreate = { acknowledge_duplication: false }): Promise<Execution> =>
    request<Execution>(`/tasks/${taskId}/executions`, { method: 'POST', ...body(payload) }),

  function: (sha256: string): Promise<FunctionRef> => request<FunctionRef>(`/functions/${sha256}`),

  providers: (): Promise<Page<Provider>> => request<Page<Provider>>('/providers'),
  provider: (id: string): Promise<Provider> => request<Provider>(`/providers/${id}`),
  createProvider: (payload: ProviderCreate): Promise<Provider> => request<Provider>('/providers', { method: 'POST', ...body(payload) }),
  updateProvider: (id: string, payload: ProviderCreate): Promise<Provider> => request<Provider>(`/providers/${id}`, { method: 'PUT', ...body(payload) }),
  deleteProvider: (id: string): Promise<void> => request<void>(`/providers/${id}`, { method: 'DELETE' }),
  providerKinds: (): Promise<ProviderKind[]> => request<ProviderKind[]>('/provider-kinds'),

  offers: (query?: OfferQuery): Promise<Page<Offer>> => request<Page<Offer>>(`/offers${qs(query)}`),
}
