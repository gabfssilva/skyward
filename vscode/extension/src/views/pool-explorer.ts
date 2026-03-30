/**
 * Pool Explorer TreeView for the Skyward sidebar.
 *
 * Flat, glanceable layout:
 *   pool → Nodes (with inline metrics), Logs
 */

import * as vscode from "vscode";
import type { NodeView, PoolView, SidecarClient } from "../sidecar/protocol";

// ── Tree node types ────────────────────────────────────────────

export type TreeNode =
  | { kind: "pool"; name: string; view: PoolView }
  | { kind: "category"; pool: string; type: "nodes" | "logs" }
  | { kind: "node"; view: NodeView; pool: string; tasksOnNode: number }
  | { kind: "metric"; label: string; value: string }
  | { kind: "bootstrap-output"; output: string };

// ── Pool Explorer ──────────────────────────────────────────────

export class PoolExplorer implements vscode.TreeDataProvider<TreeNode> {
  private readonly _onDidChangeTreeData = new vscode.EventEmitter<TreeNode | undefined>();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  private readonly _client: SidecarClient;
  private readonly _pools = new Map<string, PoolView>();

  constructor(client: SidecarClient) {
    this._client = client;
  }

  getPoolView(name: string): PoolView | undefined {
    return this._pools.get(name);
  }

  async refresh(): Promise<void> {
    const summaries = await this._client.listPools().catch(() => []);
    const views = await Promise.all(
      summaries.map((s) => this._client.getPoolView(s.name).catch(() => null)),
    );
    this._pools.clear();
    for (let i = 0; i < summaries.length; i++) {
      const view = views[i];
      if (view) this._pools.set(summaries[i].name, view);
    }
    this._onDidChangeTreeData.fire(undefined);
  }

  getTreeItem(node: TreeNode): vscode.TreeItem {
    switch (node.kind) {
      case "pool": return this._poolItem(node.name, node.view);
      case "category": return this._categoryItem(node.pool, node.type);
      case "node": return this._nodeItem(node);
      case "metric": return this._metricItem(node.label, node.value);
      case "bootstrap-output": return this._bootstrapOutputItem(node.output);
    }
  }

  getChildren(node?: TreeNode): TreeNode[] {
    if (!node) {
      return [...this._pools.entries()].map(
        ([name, view]): TreeNode => ({ kind: "pool", name, view }),
      );
    }
    if (node.kind === "pool") {
      return [
        { kind: "category", pool: node.name, type: "nodes" as const },
        { kind: "category", pool: node.name, type: "logs" as const },
      ];
    }
    if (node.kind === "category" && node.type === "nodes") {
      const view = this._pools.get(node.pool);
      if (!view) return [];
      return Object.values(view.nodes).map(
        (nv): TreeNode => ({
          kind: "node",
          view: nv,
          pool: node.pool,
          tasksOnNode: view.tasks.tasks_per_node[nv.node_id] ?? 0,
        }),
      );
    }
    if (node.kind === "node") {
      return this._nodeChildren(node.view);
    }
    return [];
  }

  // ── Pool ────────────────────────────────────────────────────

  private _poolItem(name: string, view: PoolView): vscode.TreeItem {
    const ready = Object.values(view.nodes).filter((n) => n.status === "ready").length;
    const item = new vscode.TreeItem(name, vscode.TreeItemCollapsibleState.Expanded);
    item.description = `${view.phase} · ${ready}/${view.total_nodes} nodes · $${view.cost_per_hour.toFixed(2)}/hr`;
    item.contextValue = "pool";
    item.iconPath = view.phase === "ready"
      ? new vscode.ThemeIcon("vm-active", new vscode.ThemeColor("charts.green"))
      : new vscode.ThemeIcon("loading~spin");
    item.command = { command: "skyward.showPoolDetail", title: "", arguments: [name] };
    return item;
  }

  // ── Category ────────────────────────────────────────────────

  private _categoryItem(pool: string, type: "nodes" | "logs"): vscode.TreeItem {
    if (type === "nodes") {
      const view = this._pools.get(pool);
      const count = view ? Object.keys(view.nodes).length : 0;
      const item = new vscode.TreeItem(
        `Nodes (${count})`,
        vscode.TreeItemCollapsibleState.Expanded,
      );
      item.iconPath = new vscode.ThemeIcon("server-environment");
      return item;
    }

    // Logs — click opens the Logs tab in the detail panel
    const item = new vscode.TreeItem("Logs", vscode.TreeItemCollapsibleState.None);
    item.iconPath = new vscode.ThemeIcon("output");
    item.command = { command: "skyward.showPoolLogs", title: "", arguments: [pool] };
    return item;
  }

  // ── Node ────────────────────────────────────────────────────

  private _nodeItem(node: TreeNode & { kind: "node" }): vscode.TreeItem {
    const v = node.view;
    const label = `node-${v.node_id}`;

    // Description: spec summary
    const parts: string[] = [];
    if (v.accelerator) parts.push(v.accelerator);
    if (v.ip) parts.push(v.ip);
    if (v.status === "ready") {
      parts.push(`${node.tasksOnNode} tasks`);
    } else if (v.status === "bootstrapping" && v.bootstrap) {
      parts.push(`${v.bootstrap.active} (${v.bootstrap.completed.length}/${v.bootstrap.phases.length})`);
    } else {
      parts.push(v.status);
    }

    const hasChildren =
      Object.keys(v.metrics).length > 0 ||
      (v.status === "bootstrapping" && v.bootstrap?.output);
    const collapse = hasChildren
      ? vscode.TreeItemCollapsibleState.Collapsed
      : vscode.TreeItemCollapsibleState.None;

    const item = new vscode.TreeItem(label, collapse);
    item.description = parts.join(" \u00b7 ");
    item.contextValue = "node";
    item.iconPath = this._nodeIcon(v);
    return item;
  }

  private _nodeChildren(view: NodeView): TreeNode[] {
    const children: TreeNode[] = [];

    // Metrics
    const metricLabels: Record<string, string> = {
      gpu_util: "GPU",
      vram: "VRAM",
      cpu: "CPU",
      mem: "Memory",
      gpu_temp: "GPU Temp",
    };

    for (const [key, value] of Object.entries(view.metrics)) {
      const label = metricLabels[key] ?? key;
      const isPercent = ["gpu_util", "vram", "cpu", "mem"].includes(key);
      const isTemp = key.includes("temp");
      const formatted = isPercent ? `${value.toFixed(0)}%` : isTemp ? `${value.toFixed(0)}°C` : `${value.toFixed(1)}`;
      children.push({ kind: "metric", label, value: formatted });
    }

    // Bootstrap output
    if (view.status === "bootstrapping" && view.bootstrap?.output) {
      children.push({ kind: "bootstrap-output", output: view.bootstrap.output });
    }

    return children;
  }

  private _metricItem(label: string, value: string): vscode.TreeItem {
    const item = new vscode.TreeItem(label, vscode.TreeItemCollapsibleState.None);
    item.description = value;
    item.iconPath = new vscode.ThemeIcon("dashboard");
    return item;
  }

  private _bootstrapOutputItem(output: string): vscode.TreeItem {
    const item = new vscode.TreeItem(output, vscode.TreeItemCollapsibleState.None);
    item.iconPath = new vscode.ThemeIcon("terminal");
    return item;
  }

  private _nodeIcon(view: NodeView): vscode.ThemeIcon {
    switch (view.status) {
      case "ready":
        return new vscode.ThemeIcon("circle-filled", new vscode.ThemeColor("charts.green"));
      case "bootstrapping":
        return new vscode.ThemeIcon("loading~spin", new vscode.ThemeColor("charts.yellow"));
      default:
        return new vscode.ThemeIcon("circle-outline");
    }
  }
}
