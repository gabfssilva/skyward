/**
 * Real sidecar client that spawns the Python bridge process and
 * communicates over JSON-lines (one JSON object per line on stdin/stdout).
 *
 * The bridge is started lazily on first request and kept alive for the
 * lifetime of the extension. Events (pool subscribe) arrive as unsolicited
 * JSON objects on stdout distinguished by the presence of an `event` field.
 */

import * as cp from "child_process";
import * as readline from "readline";
import type {
  EventListener,
  MainFunction,
  PoolSummary,
  PoolView,
  SidecarClient,
} from "./protocol";

type PendingRequest = {
  resolve: (value: unknown) => void;
  reject: (reason: Error) => void;
};

export class RealClient implements SidecarClient {
  private _proc: cp.ChildProcess | null = null;
  private _rl: readline.Interface | null = null;
  private _nextId = 1;
  private _pending = new Map<number, PendingRequest>();
  private _listeners = new Map<string, EventListener>();

  constructor(private _workspaceRoot: string) {}

  private _ensureProcess(): void {
    if (this._proc) return;

    this._proc = cp.spawn("uv", ["run", "python", "-m", "vscode.sidecar"], {
      cwd: this._workspaceRoot,
      stdio: ["pipe", "pipe", "pipe"],
    });

    this._rl = readline.createInterface({ input: this._proc.stdout! });
    this._rl.on("line", (line) => this._onLine(line));
    this._proc.on("exit", () => {
      this._proc = null;
      this._rl = null;
    });
  }

  private _onLine(line: string): void {
    try {
      const msg = JSON.parse(line) as Record<string, unknown>;
      if ("id" in msg) {
        const pending = this._pending.get(msg.id as number);
        if (pending) {
          this._pending.delete(msg.id as number);
          if (msg.error) pending.reject(new Error(msg.error as string));
          else pending.resolve(msg.result);
        }
      } else if ("event" in msg) {
        const pool = msg.pool as string;
        const listener = this._listeners.get(pool);
        if (listener) {
          listener(msg as unknown as Parameters<EventListener>[0]);
        }
      }
    } catch {
      /* ignore malformed lines */
    }
  }

  private _request(
    method: string,
    params: Record<string, unknown> = {},
  ): Promise<unknown> {
    this._ensureProcess();
    const id = this._nextId++;
    const line = JSON.stringify({ id, method, params });
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this._proc!.stdin!.write(line + "\n");
    });
  }

  async ping(): Promise<boolean> {
    try {
      const r = (await this._request("daemon/ping")) as { ok: boolean };
      return r.ok;
    } catch {
      return false;
    }
  }

  async startDaemon(): Promise<void> {
    await this._request("daemon/start");
  }

  async listPools(): Promise<PoolSummary[]> {
    const r = (await this._request("pools/list")) as {
      pools: PoolSummary[];
    };
    return r.pools;
  }

  async getPoolView(pool: string): Promise<PoolView> {
    return (await this._request("pools/view", { pool })) as PoolView;
  }

  async ensurePool(name: string): Promise<void> {
    await this._request("pools/ensure", { name });
  }

  async shutdownPool(pool: string): Promise<void> {
    await this._request("pools/shutdown", { pool });
  }

  subscribe(pool: string, listener: EventListener): void {
    this._listeners.set(pool, listener);
    this._request("pools/subscribe", { pool }).catch(() => {});
  }

  unsubscribe(pool: string): void {
    this._listeners.delete(pool);
    this._request("pools/unsubscribe", { pool }).catch(() => {});
  }

  async runMain(
    file: string,
    fn: string,
    args: Record<string, unknown>,
    pool: string,
  ): Promise<void> {
    await this._request("run/main", { file, fn, args, pool });
  }

  async configPools(): Promise<string[]> {
    const r = (await this._request("config/pools")) as { pools: string[] };
    return r.pools;
  }

  async configProviders(): Promise<string[]> {
    const r = (await this._request("config/providers")) as { providers: string[] };
    return r.providers;
  }

  async discoverMainFunctions(files?: string[]): Promise<MainFunction[]> {
    const r = (await this._request("discover/functions", { files })) as {
      functions: MainFunction[];
    };
    return r.functions;
  }

  dispose(): void {
    if (this._proc) {
      this._proc.kill();
      this._proc = null;
    }
    this._rl = null;
    this._pending.clear();
    this._listeners.clear();
  }
}
