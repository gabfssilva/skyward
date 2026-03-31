/**
 * Mock implementation of {@link SidecarClient} for UI development.
 *
 * Returns hardcoded data so that TreeView, StatusBar, and CodeLens can
 * be validated without a running Skyward daemon.
 */

import type {
  EventListener,
  MainFunction,
  PoolSummary,
  PoolView,
  SidecarClient,
  SkywardEvent,
} from "./protocol";

// ── Helpers ─────────────────────────────────────────────────────

function randomChoice<T>(items: T[]): T {
  return items[Math.floor(Math.random() * items.length)];
}

function mockTaskId(): string {
  return `task-${Date.now()}-${Math.floor(Math.random() * 1000)}`;
}

// ── Mock pool view ──────────────────────────────────────────────

function buildMockPoolView(name: string): PoolView {
  return {
    name,
    phase: "ready",
    total_nodes: 4,
    started_at: Date.now() / 1000 - 300,
    ready_at: Date.now() / 1000 - 120,
    cost_per_hour: 13.2,
    cost_total: 6.6,
    nodes: {
      0: {
        node_id: 0,
        status: "ready",
        ip: "10.0.1.10",
        accelerator: "A100-80GB",
        metrics: { cpu: 34, mem: 48, gpu_util: 87, vram: 78, gpu_temp: 71 },
      },
      1: {
        node_id: 1,
        status: "ready",
        ip: "10.0.1.11",
        accelerator: "A100-80GB",
        metrics: { cpu: 41, mem: 52, gpu_util: 92, vram: 89, gpu_temp: 74 },
      },
      2: {
        node_id: 2,
        status: "ready",
        ip: "10.0.1.12",
        accelerator: "A100-80GB",
        metrics: { cpu: 28, mem: 35, gpu_util: 45, vram: 40, gpu_temp: 63 },
      },
      3: {
        node_id: 3,
        status: "bootstrapping",
        accelerator: "A100-80GB",
        bootstrap: {
          phases: ["connecting", "apt", "uv", "deps", "worker"],
          completed: ["connecting", "apt"],
          active: "uv",
          output: "Installing dependencies...",
        },
        metrics: {},
      },
    },
    logs: [
      { ts: Date.now() / 1000 - 180, level: "info", node_id: 0, message: "Worker ready, listening on port 9100" },
      { ts: Date.now() / 1000 - 120, level: "info", node_id: 1, message: "Worker ready, listening on port 9100" },
      { ts: Date.now() / 1000 - 90, level: "info", node_id: 0, message: "Epoch 1/50 — loss=0.423, lr=0.001" },
      { ts: Date.now() / 1000 - 75, level: "info", node_id: 1, message: "Epoch 1/50 — loss=0.418, lr=0.001" },
      { ts: Date.now() / 1000 - 60, level: "info", node_id: 0, message: "Epoch 2/50 — loss=0.312, lr=0.001" },
      { ts: Date.now() / 1000 - 55, level: "info", node_id: 2, message: "Worker ready, listening on port 9100" },
      { ts: Date.now() / 1000 - 45, level: "info", node_id: 1, message: "Epoch 2/50 — loss=0.305, lr=0.001" },
      { ts: Date.now() / 1000 - 30, level: "info", node_id: 0, message: "Checkpoint saved to /tmp/ckpt-002.pt" },
      { ts: Date.now() / 1000 - 20, level: "warn", node_id: 2, message: "GPU memory usage above 90%" },
      { ts: Date.now() / 1000 - 10, level: "info", node_id: 0, message: "Epoch 3/50 — loss=0.248, lr=0.0008" },
      { ts: Date.now() / 1000 - 5, level: "info", node_id: 1, message: "Epoch 3/50 — loss=0.241, lr=0.0008" },
      { ts: Date.now() / 1000 - 2, level: "error", node_id: 3, message: "Bootstrap failed: uv install timeout after 120s" },
    ],
    tasks: {
      queued: 3,
      running: 2,
      done: 47,
      failed: 1,
      inflight: {
        "task-001": {
          task_id: "task-001",
          name: "train(batch=64)",
          kind: "rshift",
          started_at: Date.now() / 1000 - 12,
          node_id: 0,
          broadcast_total: 0,
          broadcast_done: 0,
        },
        "task-002": {
          task_id: "task-002",
          name: "evaluate(split='val')",
          kind: "rshift",
          started_at: Date.now() / 1000 - 5,
          node_id: 1,
          broadcast_total: 0,
          broadcast_done: 0,
        },
      },
      throughput: 12.3,
      avg_latency: 1.04,
      tasks_per_node: { 0: 16, 1: 18, 2: 13 },
      fn_summary: {
        "train": { calls: 10, avg: 1.07, min: 0.7, max: 1.5, failed: 1 },
        "evaluate": { calls: 7, avg: 0.41, min: 0.35, max: 0.5, failed: 0 },
        "preprocess": { calls: 5, avg: 0.12, min: 0.10, max: 0.14, failed: 0 },
      },
      first_task_at: Date.now() / 1000 - 240,
    },
    scaling: {
      desired: 4,
      pending: 1,
      draining: 0,
      reconciler_state: "scaling_up",
      is_elastic: true,
      min_nodes: 2,
      max_nodes: 8,
    },
  };
}

// ── Random event emitters ───────────────────────────────────────

function randomEvent(pool: string): SkywardEvent {
  const kind = randomChoice([
    "task.completed",
    "task.completed",
    "metric.sampled",
    "metric.sampled",
    "log.emitted",
  ]);

  switch (kind) {
    case "task.completed":
      return {
        event: "task.completed",
        pool,
        data: {
          task_id: mockTaskId(),
          node_id: randomChoice([0, 1, 2]),
          elapsed: Math.random() * 2 + 0.3,
        },
      };
    case "metric.sampled":
      return {
        event: "metric.sampled",
        pool,
        data: {
          node_id: randomChoice([0, 1, 2]),
          name: randomChoice(["gpu_util", "mem_used_gb", "temperature"]),
          value: Math.round(Math.random() * 100),
        },
      };
    case "log.emitted":
      return {
        event: "log.emitted",
        pool,
        data: {
          node_id: randomChoice([0, 1, 2, 3]),
          message: randomChoice([
            "Epoch 12/50 completed — loss=0.023",
            "Checkpoint saved to /tmp/ckpt-12.pt",
            "Gradient sync across 3 nodes took 45ms",
            "Batch 384/1024 processed",
          ]),
        },
      };
    default:
      return { event: kind, pool, data: {} };
  }
}

// ── MockClient ──────────────────────────────────────────────────

export class MockClient implements SidecarClient {
  private readonly _intervals = new Map<string, ReturnType<typeof setInterval>>();
  private readonly _listeners = new Map<string, EventListener>();
  private readonly _timeouts: ReturnType<typeof setTimeout>[] = [];

  async ping(): Promise<boolean> {
    return true;
  }

  async startDaemon(): Promise<void> {
    /* no-op */
  }

  async listPools(): Promise<PoolSummary[]> {
    return [
      { name: "train", phase: "ready", ready_nodes: 3, total_nodes: 4 },
      {
        name: "inference",
        phase: "provisioning",
        ready_nodes: 0,
        total_nodes: 2,
      },
    ];
  }

  async getPoolView(pool: string): Promise<PoolView> {
    return buildMockPoolView(pool);
  }

  async ensurePool(_name: string): Promise<void> {
    /* no-op */
  }

  async shutdownPool(_pool: string): Promise<void> {
    /* no-op */
  }

  subscribe(pool: string, listener: EventListener): void {
    this.unsubscribe(pool);
    this._listeners.set(pool, listener);

    const interval = setInterval(() => {
      const cb = this._listeners.get(pool);
      if (cb) {
        cb(randomEvent(pool));
      }
    }, 3_000);

    this._intervals.set(pool, interval);
  }

  unsubscribe(pool: string): void {
    const interval = this._intervals.get(pool);
    if (interval) {
      clearInterval(interval);
      this._intervals.delete(pool);
    }
    this._listeners.delete(pool);
  }

  async runMain(
    file: string,
    fn: string,
    _args: Record<string, unknown>,
    pool: string,
  ): Promise<void> {
    const taskId = mockTaskId();
    const listener = this._listeners.get(pool);
    if (!listener) {
      return;
    }

    listener({
      event: "task.queued",
      pool,
      data: { task_id: taskId, name: `${fn}()`, file },
    });

    const t1 = setTimeout(() => {
      listener({
        event: "task.assigned",
        pool,
        data: { task_id: taskId, node_id: 0 },
      });
    }, 500);
    this._timeouts.push(t1);

    const t2 = setTimeout(() => {
      listener({
        event: "task.completed",
        pool,
        data: { task_id: taskId, node_id: 0, elapsed: 2.5 },
      });
    }, 3_000);
    this._timeouts.push(t2);
  }

  async configPools(): Promise<string[]> {
    return ["train", "inference", "eval"];
  }

  async configProviders(): Promise<string[]> {
    return ["my-aws", "cheap-vastai"];
  }

  async discoverMainFunctions(_files?: string[]): Promise<MainFunction[]> {
    return [
      {
        name: "train",
        file: "train.py",
        line: 12,
        params: [
          { name: "epochs", type: "int", default: 50 },
          { name: "lr", type: "float", default: 0.001 },
          {
            name: "backend",
            type: "str",
            default: "nccl",
            choices: ["nccl", "gloo"],
          },
        ],
      },
      {
        name: "evaluate",
        file: "eval.py",
        line: 8,
        params: [
          { name: "split", type: "str", default: "test" },
          { name: "verbose", type: "bool", default: false },
        ],
      },
    ];
  }

  dispose(): void {
    for (const interval of this._intervals.values()) {
      clearInterval(interval);
    }
    this._intervals.clear();

    for (const timeout of this._timeouts) {
      clearTimeout(timeout);
    }
    this._timeouts.length = 0;

    this._listeners.clear();
  }
}
