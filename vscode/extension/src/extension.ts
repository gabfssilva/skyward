import * as vscode from "vscode";
import { registerLifecycleCommands } from "./commands/lifecycle";
import { SkyMainCodeLensProvider } from "./codelens/main-lens";
import type { ParsedParam } from "./codelens/main-lens";
import { NodeLogManager } from "./logs/node-channel";
import { createClient } from "./sidecar/client";
import type { SkywardEvent } from "./sidecar/protocol";
import { PoolStatusBar } from "./statusbar/pool-status";
import { DetailPanelManager } from "./views/detail-panel";
import { LogPanelManager } from "./views/log-panel";
import { PoolExplorer } from "./views/pool-explorer";
import type { TreeNode } from "./views/pool-explorer";

// ── Helpers ──────────────────────────────────────────────────────

function debounce(fn: () => void, ms: number): () => void {
  let timer: ReturnType<typeof setTimeout> | undefined;
  return () => {
    if (timer) clearTimeout(timer);
    timer = setTimeout(fn, ms);
  };
}

// ── Activate ─────────────────────────────────────────────────────

export function activate(context: vscode.ExtensionContext): void {
  const useMock = vscode.workspace
    .getConfiguration("skyward")
    .get("useMock", true);
  const workspaceRoot =
    vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? process.cwd();
  const client = createClient(workspaceRoot, useMock);
  const statusBar = new PoolStatusBar(client);
  const poolExplorer = new PoolExplorer(client);
  const codeLensProvider = new SkyMainCodeLensProvider();
  const logManager = new NodeLogManager();
  const detailPanel = new DetailPanelManager();
  const logPanel = new LogPanelManager();

  context.subscriptions.push(client);
  context.subscriptions.push(statusBar);
  context.subscriptions.push(logManager);
  context.subscriptions.push(detailPanel);
  context.subscriptions.push(logPanel);

  // ── Views ────────────────────────────────────────────────────

  context.subscriptions.push(
    vscode.window.registerTreeDataProvider("skyward.poolExplorer", poolExplorer),
  );

  context.subscriptions.push(
    vscode.languages.registerCodeLensProvider({ language: "python" }, codeLensProvider),
  );

  // ── Commands ─────────────────────────────────────────────────

  context.subscriptions.push(
    vscode.commands.registerCommand("skyward.selectPool", async () => {
      const names = statusBar.getPoolNames();
      if (names.length === 0) {
        vscode.window.showWarningMessage("No Skyward pools available.");
        return;
      }

      const picked = await vscode.window.showQuickPick(names, {
        placeHolder: "Select active pool",
      });

      if (picked) {
        statusBar.selectPool(picked);
        codeLensProvider.setActivePool(picked);
      }
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("skyward.startDaemon", async () => {
      await client.startDaemon();
      vscode.window.showInformationMessage("Skyward daemon started.");
      await statusBar.refresh();
    }),
  );

  const syncCodeLens = (): void => {
    const names = statusBar.getPoolNames();
    codeLensProvider.setPoolCount(names.length);
    const active = statusBar.active;
    if (active) codeLensProvider.setActivePool(active);
  };

  const refreshAll = (): void => {
    void poolExplorer.refresh();
    void statusBar.refresh().then(() => {
      syncCodeLens();
      subscribeAll();
    });
  };

  registerLifecycleCommands(context, client, statusBar, poolExplorer, refreshAll);

  context.subscriptions.push(
    vscode.commands.registerCommand("skyward.refreshPools", async () => {
      await poolExplorer.refresh();
      await statusBar.refresh();
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand(
      "skyward.showNodeLogs",
      (node?: TreeNode) => {
        if (node?.kind === "node") {
          logManager.showLogs(node.pool, node.view.node_id);
        }
      },
    ),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand(
      "skyward.runMain",
      async (uri: vscode.Uri, fnName: string, params: ParsedParam[]) => {
        const args: Record<string, unknown> = {};

        for (const param of params) {
          const input = await vscode.window.showInputBox({
            prompt: `${param.name} (${param.type})`,
            value: param.default,
          });

          if (input === undefined) {
            return;
          }

          args[param.name] = parseValue(input, param.type);
        }

        const poolNames = statusBar.getPoolNames();
        let pool: string | undefined;

        if (poolNames.length === 0) {
          vscode.window.showWarningMessage("No active pools. Start one first.");
          return;
        } else if (poolNames.length === 1) {
          pool = poolNames[0];
        } else {
          pool = await vscode.window.showQuickPick(poolNames, {
            placeHolder: "Select pool to run on",
          });
        }

        if (!pool) return;
        const argStr = Object.entries(args)
          .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
          .join(", ");
        vscode.window.showInformationMessage(
          `Running ${fnName}(${argStr}) on ${pool}...`,
        );

        try {
          await client.runMain(uri.fsPath, fnName, args, pool);
          vscode.window.showInformationMessage(
            `${fnName}(${argStr}) completed on ${pool}.`,
          );
        } catch (err) {
          vscode.window.showErrorMessage(
            `${fnName} failed: ${err instanceof Error ? err.message : err}`,
          );
        }
      },
    ),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("skyward.showPoolDetail", (poolName: string) => {
      const view = poolExplorer.getPoolView(poolName);
      if (view) {
        detailPanel.showPoolDetail(view);
      }
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("skyward.showPoolLogs", (poolName: string) => {
      const view = poolExplorer.getPoolView(poolName);
      if (view) {
        logPanel.showLogs(view);
      }
    }),
  );

  // ── Event stream ─────────────────────────────────────────────

  const debouncedRefresh = debounce(() => {
    void poolExplorer.refresh();
    void statusBar.refresh().then(() => {
      syncCodeLens();
      subscribeAll();
    });
  }, 500);

  const handleEvent = (event: SkywardEvent): void => {
    logManager.handleEvent(event);
    const getView = (pool: string) => poolExplorer.getPoolView(pool);
    detailPanel.handleEvent(event, getView);
    logPanel.handleEvent(event, getView);
    debouncedRefresh();
  };

  // Subscribe to pool events — tracks which pools are already subscribed
  const subscribedPools = new Set<string>();

  const subscribeAll = (): void => {
    void client.listPools().then((pools) => {
      for (const pool of pools) {
        if (!subscribedPools.has(pool.name)) {
          subscribedPools.add(pool.name);
          client.subscribe(pool.name, handleEvent);
        }
      }
    });
  };

  // ── Initial load ─────────────────────────────────────────────

  void poolExplorer.refresh();
  void statusBar.refresh().then(() => {
    syncCodeLens();
    subscribeAll();
  });
}

function parseValue(raw: string, type: string): unknown {
  switch (type) {
    case "int":
      return parseInt(raw, 10);
    case "float":
      return parseFloat(raw);
    case "bool":
      return raw.toLowerCase() === "true";
    default:
      return raw;
  }
}

export function deactivate(): void {}
