/**
 * Pool Explorer TreeView for the Skyward sidebar.
 *
 * Shows running pools (from daemon) + configured-but-stopped pools
 * (from skyward.toml) in a unified tree.
 */

import * as vscode from "vscode";
import type { NodeView, PoolPhase, PoolView, SidecarClient } from "../sidecar/protocol";

// ── Tree node types ────────────────────────────────────────────

export type TreeNode =
  | { kind: "pool"; name: string; view: PoolView }
  | { kind: "stopped-pool"; name: string }
  | { kind: "starting-pool"; name: string }
  | { kind: "phase-step"; label: string; status: "done" | "active" | "pending"; detail?: string }
  | { kind: "category"; pool: string; type: "nodes" | "logs" }
  | { kind: "node"; view: NodeView; pool: string; tasksOnNode: number }
  | { kind: "metric"; label: string; value: string }
  | { kind: "bootstrap-output"; output: string };

// ── Lifecycle phases ──────────────────────────────────────────

const LIFECYCLE_PHASES: { phase: PoolPhase; label: string }[] = [
  { phase: "provisioning", label: "Provision instances" },
  { phase: "ssh", label: "Connect via SSH" },
  { phase: "bootstrap", label: "Bootstrap environment" },
  { phase: "workers", label: "Start workers" },
  { phase: "ready", label: "Ready" },
];

function phaseIndex(phase: PoolPhase): number {
  const idx = LIFECYCLE_PHASES.findIndex((p) => p.phase === phase);
  return idx >= 0 ? idx : 0;
}

function buildPhaseSteps(view: PoolView): TreeNode[] {
  const current = phaseIndex(view.phase);
  const nodes = Object.values(view.nodes);
  const total = view.total_nodes;

  return LIFECYCLE_PHASES.map((lp, i): TreeNode => {
    const status = i < current ? "done" : i === current ? "active" : "pending";
    let detail: string | undefined;

    if (i === current) {
      switch (lp.phase) {
        case "provisioning":
          detail = nodes.length > 0 ? `${nodes.length}/${total}` : undefined;
          break;
        case "ssh": {
          const connected = nodes.filter((n) => n.status !== "waiting").length;
          detail = `${connected}/${total}`;
          break;
        }
        case "bootstrap": {
          // Show the active bootstrap phase from the first bootstrapping node
          const bootstrapping = nodes.find((n) => n.status === "bootstrapping" && n.bootstrap);
          if (bootstrapping?.bootstrap) {
            const bs = bootstrapping.bootstrap;
            detail = `${bs.completed.length}/${bs.phases.length} · ${bs.active}`;
          } else {
            const done = nodes.filter((n) => n.status === "ready").length;
            detail = `${done}/${total} nodes`;
          }
          break;
        }
        case "workers": {
          const ready = nodes.filter((n) => n.status === "ready").length;
          detail = `${ready}/${total}`;
          break;
        }
      }
    }

    return { kind: "phase-step", label: lp.label, status, detail };
  });
}

// ── Pool Explorer ──────────────────────────────────────────────

export class PoolExplorer implements vscode.TreeDataProvider<TreeNode> {
  private readonly _onDidChangeTreeData = new vscode.EventEmitter<TreeNode | undefined>();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  private readonly _client: SidecarClient;
  private readonly _pools = new Map<string, PoolView>();
  private _configPoolNames: string[] = [];
  private readonly _startingPools = new Set<string>();

  constructor(client: SidecarClient) {
    this._client = client;
  }

  getPoolView(name: string): PoolView | undefined {
    return this._pools.get(name);
  }

  markStarting(name: string): void {
    this._startingPools.add(name);
    this._onDidChangeTreeData.fire(undefined);
  }

  async refresh(): Promise<void> {
    const [summaries, configNames] = await Promise.all([
      this._client.listPools().catch(() => []),
      this._client.configPools().catch(() => []),
    ]);

    const views = await Promise.all(
      summaries.map((s) => this._client.getPoolView(s.name).catch(() => null)),
    );
    this._pools.clear();
    for (let i = 0; i < summaries.length; i++) {
      const view = views[i];
      if (view) {
        this._pools.set(summaries[i].name, view);
        // Only clear "starting" once the pool has actually moved past stopped
        if (view.phase !== "stopped") {
          this._startingPools.delete(summaries[i].name);
        }
      }
    }
    this._configPoolNames = configNames;
    this._onDidChangeTreeData.fire(undefined);
  }

  getTreeItem(node: TreeNode): vscode.TreeItem {
    switch (node.kind) {
      case "pool": return this._poolItem(node.name, node.view);
      case "stopped-pool": return this._stoppedPoolItem(node.name);
      case "starting-pool": return this._startingPoolItem(node.name);
      case "phase-step": return this._phaseStepItem(node);
      case "category": return this._categoryItem(node.pool, node.type);
      case "node": return this._nodeItem(node);
      case "metric": return this._metricItem(node.label, node.value);
      case "bootstrap-output": return this._bootstrapOutputItem(node.output);
    }
  }

  getChildren(node?: TreeNode): TreeNode[] {
    if (!node) {
      const children: TreeNode[] = [];

      for (const [name, view] of this._pools) {
        if (this._startingPools.has(name) && view.phase === "stopped") {
          children.push({ kind: "starting-pool", name });
        } else {
          children.push({ kind: "pool", name, view });
        }
      }

      for (const name of this._startingPools) {
        if (!this._pools.has(name)) {
          children.push({ kind: "starting-pool", name });
        }
      }

      for (const name of this._configPoolNames) {
        if (!this._pools.has(name) && !this._startingPools.has(name)) {
          children.push({ kind: "stopped-pool", name });
        }
      }

      return children;
    }

    if (node.kind === "pool") {
      // Stopped pools show no children — just the pool item itself
      if (node.view.phase === "stopped") return [];

      const children: TreeNode[] = [];

      // Show lifecycle checklist only during active startup phases
      if (node.view.phase !== "ready") {
        children.push(...buildPhaseSteps(node.view));
      }

      children.push({ kind: "category", pool: node.name, type: "nodes" as const });
      children.push({ kind: "category", pool: node.name, type: "logs" as const });
      return children;
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
    const total = view.total_nodes;
    const nodes = Object.values(view.nodes);
    const ready = nodes.filter((n) => n.status === "ready").length;
    const cost = view.cost_per_hour > 0 ? ` · $${view.cost_per_hour.toFixed(2)}/hr` : "";

    let description: string;
    let icon: vscode.ThemeIcon;

    switch (view.phase) {
      case "provisioning":
        description = `provisioning instances...${cost}`;
        icon = new vscode.ThemeIcon("loading~spin", new vscode.ThemeColor("charts.yellow"));
        break;
      case "ssh": {
        const connected = nodes.filter((n) => n.status !== "waiting").length;
        description = `connecting via SSH · ${connected}/${total}${cost}`;
        icon = new vscode.ThemeIcon("loading~spin", new vscode.ThemeColor("charts.yellow"));
        break;
      }
      case "bootstrap": {
        const bootstrapped = nodes.filter((n) => n.status === "bootstrapping" || n.status === "ready").length;
        description = `bootstrapping · ${bootstrapped}/${total} ready${cost}`;
        icon = new vscode.ThemeIcon("loading~spin", new vscode.ThemeColor("charts.yellow"));
        break;
      }
      case "workers":
        description = `starting workers · ${ready}/${total}${cost}`;
        icon = new vscode.ThemeIcon("loading~spin", new vscode.ThemeColor("charts.yellow"));
        break;
      case "stopped": {
        const item = new vscode.TreeItem(name, vscode.TreeItemCollapsibleState.None);
        item.description = "click to start";
        item.contextValue = "stopped-pool";
        item.iconPath = new vscode.ThemeIcon("debug-start", new vscode.ThemeColor("descriptionForeground"));
        item.tooltip = `Start pool "${name}"`;
        item.command = { command: "skyward.startPool", title: "", arguments: [name] };
        return item;
      }
      case "ready":
      default:
        description = `${ready}/${total} nodes${cost}`;
        icon = new vscode.ThemeIcon("vm-active", new vscode.ThemeColor("charts.green"));
        break;
    }

    const item = new vscode.TreeItem(name, vscode.TreeItemCollapsibleState.Expanded);
    item.description = description;
    item.contextValue = "pool";
    item.iconPath = icon;
    item.command = { command: "skyward.showPoolDetail", title: "", arguments: [name] };
    return item;
  }

  // ── Starting Pool ────────────────────────────────────────

  private _startingPoolItem(name: string): vscode.TreeItem {
    const item = new vscode.TreeItem(name, vscode.TreeItemCollapsibleState.None);
    item.description = "launching...";
    item.iconPath = new vscode.ThemeIcon("loading~spin", new vscode.ThemeColor("charts.yellow"));
    item.tooltip = `Pool "${name}" is starting up`;
    return item;
  }

  // ── Stopped Pool ───────────────────────────────────────────

  private _stoppedPoolItem(name: string): vscode.TreeItem {
    const item = new vscode.TreeItem(name, vscode.TreeItemCollapsibleState.None);
    item.description = "click to start";
    item.contextValue = "stopped-pool";
    item.iconPath = new vscode.ThemeIcon("debug-start", new vscode.ThemeColor("descriptionForeground"));
    item.tooltip = `Start pool "${name}"`;
    item.command = { command: "skyward.startPool", title: "", arguments: [name] };
    return item;
  }

  // ── Phase Step ─────────────────────────────────────────────

  private _phaseStepItem(node: TreeNode & { kind: "phase-step" }): vscode.TreeItem {
    const item = new vscode.TreeItem(node.label, vscode.TreeItemCollapsibleState.None);

    switch (node.status) {
      case "done":
        item.iconPath = new vscode.ThemeIcon("pass-filled", new vscode.ThemeColor("charts.green"));
        break;
      case "active":
        item.iconPath = new vscode.ThemeIcon("loading~spin", new vscode.ThemeColor("charts.yellow"));
        break;
      case "pending":
        item.iconPath = new vscode.ThemeIcon("circle-outline", new vscode.ThemeColor("descriptionForeground"));
        break;
    }

    if (node.detail) {
      item.description = node.detail;
    }

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

    const item = new vscode.TreeItem("Logs", vscode.TreeItemCollapsibleState.None);
    item.iconPath = new vscode.ThemeIcon("output");
    item.command = { command: "skyward.showPoolLogs", title: "", arguments: [pool] };
    return item;
  }

  // ── Node ────────────────────────────────────────────────────

  private _nodeItem(node: TreeNode & { kind: "node" }): vscode.TreeItem {
    const v = node.view;
    const label = `node-${v.node_id}`;

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
