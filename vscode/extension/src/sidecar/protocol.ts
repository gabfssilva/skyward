/**
 * Shared TypeScript types mirroring the Skyward Python view hierarchy.
 *
 * These types are the contract between the VS Code extension UI layer
 * (TreeView, StatusBar, CodeLens) and the sidecar data layer (daemon
 * client or mock). They correspond to `skyward.api.views` on the
 * Python side.
 */

// ── Node ────────────────────────────────────────────────────────

export type NodeStatus = "waiting" | "ssh" | "bootstrapping" | "ready";

export interface BootstrapView {
  phases: string[];
  completed: string[];
  active: string;
  output: string;
}

export interface NodeView {
  node_id: number;
  status: NodeStatus;
  ip?: string;
  accelerator?: string;
  bootstrap?: BootstrapView;
  metrics: Record<string, number>;
}

// ── Tasks ───────────────────────────────────────────────────────

export interface TaskEntry {
  task_id: string;
  name: string;
  kind: string;
  started_at: number;
  node_id: number;
  broadcast_total: number;
  broadcast_done: number;
}

export interface FnSummary {
  calls: number;
  avg: number;
  min: number;
  max: number;
  failed: number;
}

export interface TasksView {
  queued: number;
  running: number;
  done: number;
  failed: number;
  inflight: Record<string, TaskEntry>;
  throughput: number;
  avg_latency: number;
  tasks_per_node: Record<number, number>;
  fn_summary: Record<string, FnSummary>;
  first_task_at: number;
}

// ── Scaling ─────────────────────────────────────────────────────

export interface ScalingView {
  desired: number;
  pending: number;
  draining: number;
  reconciler_state: string;
  is_elastic: boolean;
  min_nodes?: number;
  max_nodes?: number;
}

// ── Pool ────────────────────────────────────────────────────────

export type PoolPhase =
  | "provisioning"
  | "ssh"
  | "bootstrap"
  | "workers"
  | "ready"
  | "stopped";

export interface PoolView {
  name: string;
  phase: PoolPhase;
  total_nodes: number;
  nodes: Record<number, NodeView>;
  tasks: TasksView;
  scaling: ScalingView;
  started_at: number;
  ready_at: number;
  cost_per_hour: number;
  cost_total: number;
  logs: LogEntry[];
}

export interface PoolSummary {
  name: string;
  phase: PoolPhase;
  ready_nodes: number;
  total_nodes: number;
}

// ── Events ──────────────────────────────────────────────────────

export interface SkywardEvent {
  event: string;
  pool: string;
  data: Record<string, unknown>;
}

// ── Main function discovery ─────────────────────────────────────

export interface MainFunctionParam {
  name: string;
  type: "int" | "float" | "str" | "bool";
  default?: string | number | boolean;
  choices?: string[];
}

export interface MainFunction {
  name: string;
  file: string;
  line: number;
  params: MainFunctionParam[];
}

// ── Log entries ────────────────────────────────────────────────

export interface LogEntry {
  ts: number;
  level: string;
  node_id?: number;
  message: string;
}

// ── Client interface ────────────────────────────────────────────

export type EventListener = (event: SkywardEvent) => void;

export interface SidecarClient {
  ping(): Promise<boolean>;
  startDaemon(): Promise<void>;
  listPools(): Promise<PoolSummary[]>;
  getPoolView(pool: string): Promise<PoolView>;
  ensurePool(name: string): Promise<void>;
  shutdownPool(pool: string): Promise<void>;
  subscribe(pool: string, listener: EventListener): void;
  unsubscribe(pool: string): void;
  runMain(
    file: string,
    fn: string,
    args: Record<string, unknown>,
    pool: string,
  ): Promise<void>;
  configPools(): Promise<string[]>;
  configProviders(): Promise<string[]>;
  discoverMainFunctions(files?: string[]): Promise<MainFunction[]>;
  dispose(): void;
}
