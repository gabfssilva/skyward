/**
 * Status bar component showing the active Skyward pool and node counts.
 *
 * Displays an icon with the pool name and ready/total node counts.
 * Clicking the item opens the pool selector quick pick. When the daemon
 * is unreachable the item switches to an offline state with a command
 * to start it.
 */

import * as vscode from "vscode";
import type { PoolSummary, SidecarClient, SkywardEvent } from "../sidecar/protocol";

export class PoolStatusBar implements vscode.Disposable {
  private readonly _item: vscode.StatusBarItem;
  private readonly _client: SidecarClient;
  private _activePool: string | undefined;
  private _pools: PoolSummary[] = [];

  constructor(client: SidecarClient) {
    this._client = client;
    this._item = vscode.window.createStatusBarItem(
      vscode.StatusBarAlignment.Left,
      100,
    );
    this._item.command = "skyward.selectPool";
    this._item.show();
  }

  get active(): string | undefined {
    return this._activePool;
  }

  async refresh(): Promise<void> {
    const alive = await this._client.ping().catch(() => false);
    if (!alive) {
      this.setOffline();
      return;
    }

    this._pools = await this._client.listPools().catch(() => []);

    if (!this._activePool && this._pools.length > 0) {
      this._activePool = this._pools[0].name;
    }

    if (this._activePool && !this._pools.find((p) => p.name === this._activePool)) {
      this._activePool = this._pools.length > 0 ? this._pools[0].name : undefined;
    }

    this.render();
  }

  selectPool(name: string): void {
    this._activePool = name;
    this.render();
  }

  handleEvent(_event: SkywardEvent): void {
    void this.refresh();
  }

  getPoolNames(): string[] {
    return this._pools.map((p) => p.name);
  }

  dispose(): void {
    this._item.dispose();
  }

  private render(): void {
    const pool = this._pools.find((p) => p.name === this._activePool);
    if (!pool) {
      this._item.text = "$(vm-active) Skyward: no pools";
      this._item.backgroundColor = undefined;
      this._item.tooltip = "No active pools";
      this._item.command = "skyward.selectPool";
      return;
    }

    const icon = pool.phase === "ready" ? "$(vm-active)" : "$(loading~spin)";
    this._item.text = `${icon} ${pool.name} [${pool.ready_nodes}/${pool.total_nodes}]`;
    this._item.backgroundColor = undefined;
    this._item.tooltip = `Pool: ${pool.name} (${pool.phase}) - ${pool.ready_nodes} ready / ${pool.total_nodes} total`;
    this._item.command = "skyward.selectPool";
  }

  private setOffline(): void {
    this._item.text = "$(circle-slash) Skyward: offline";
    this._item.backgroundColor = new vscode.ThemeColor(
      "statusBarItem.warningBackground",
    );
    this._item.tooltip = "Daemon not running. Click to start.";
    this._item.command = "skyward.startDaemon";
  }
}
