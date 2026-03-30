/**
 * Per-node OutputChannel log manager.
 *
 * Opens a VS Code OutputChannel for each node the user requests logs for,
 * then routes incoming {@link SkywardEvent}s to the matching channel with
 * human-readable formatting.
 */

import * as vscode from "vscode";
import type { SkywardEvent } from "../sidecar/protocol";

export class NodeLogManager implements vscode.Disposable {
  private readonly _channels = new Map<string, vscode.OutputChannel>();

  /**
   * Open (or focus) the log channel for a specific node.
   */
  showLogs(pool: string, nodeId: number): void {
    const key = `${pool}/node-${nodeId}`;

    if (!this._channels.has(key)) {
      const channel = vscode.window.createOutputChannel(`Skyward: ${key}`);
      this._channels.set(key, channel);
    }

    this._channels.get(key)!.show(true);
  }

  /**
   * Route an event to the appropriate node channel.
   *
   * If no channel is open for the event's node, the event is silently
   * dropped. Metric samples are always skipped to avoid log noise.
   */
  handleEvent(event: SkywardEvent): void {
    if (event.event === "metric.sampled") {
      return;
    }

    const nodeId = event.data.node_id;
    if (nodeId === undefined) {
      return;
    }

    const key = `${event.pool}/node-${nodeId}`;
    const channel = this._channels.get(key);
    if (!channel) {
      return;
    }

    const line = this._format(event);
    if (line) {
      channel.appendLine(line);
    }
  }

  /**
   * Dispose and remove all channels belonging to a pool.
   */
  clearPool(pool: string): void {
    for (const [key, channel] of this._channels) {
      if (key.startsWith(`${pool}/`)) {
        channel.dispose();
        this._channels.delete(key);
      }
    }
  }

  /**
   * Dispose all channels.
   */
  dispose(): void {
    for (const channel of this._channels.values()) {
      channel.dispose();
    }
    this._channels.clear();
  }

  // ── Formatting ─────────────────────────────────────────────────

  private _format(event: SkywardEvent): string | null {
    const d = event.data;

    switch (event.event) {
      case "node.bootstrap.output":
        return `[bootstrap] ${d.output ?? ""}`;

      case "log.emitted":
        return `[${d.level ?? "info"}] ${d.message ?? ""}`;

      case "task.assigned":
        return `[task:${d.task_id}] assigned`;

      case "task.completed": {
        const elapsed =
          typeof d.elapsed === "number" ? d.elapsed.toFixed(2) : String(d.elapsed);
        return `[task:${d.task_id}] completed in ${elapsed}s`;
      }

      case "task.failed":
        return `[task:${d.task_id}] FAILED: ${d.error ?? "unknown"}`;

      default:
        return null;
    }
  }
}
