/**
 * Pool lifecycle commands: start, stop, and inline-stop.
 *
 * Registers three commands that drive pool lifecycle through the sidecar
 * client and keep the status bar and explorer in sync.
 */

import * as vscode from "vscode";
import type { SidecarClient } from "../sidecar/protocol";
import type { PoolStatusBar } from "../statusbar/pool-status";
import type { TreeNode } from "../views/pool-explorer";

export function registerLifecycleCommands(
  context: vscode.ExtensionContext,
  client: SidecarClient,
  statusBar: PoolStatusBar,
  onPoolChanged: () => void,
): void {
  context.subscriptions.push(
    vscode.commands.registerCommand("skyward.startPool", async () => {
      const poolNames = ["train", "inference", "eval"];

      const picked = await vscode.window.showQuickPick(poolNames, {
        placeHolder: "Select a pool to start",
      });

      if (!picked) {
        return;
      }

      await vscode.window.withProgress(
        { location: vscode.ProgressLocation.Notification, title: "Starting pool..." },
        async () => {
          await client.ensurePool(picked);
        },
      );

      vscode.window.showInformationMessage(`Pool "${picked}" started.`);
      statusBar.selectPool(picked);
      onPoolChanged();
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("skyward.stopPool", async () => {
      const pools = await client.listPools();

      if (pools.length === 0) {
        vscode.window.showWarningMessage("No active pools to stop.");
        return;
      }

      const picked = await vscode.window.showQuickPick(
        pools.map((p) => p.name),
        { placeHolder: "Select a pool to stop" },
      );

      if (!picked) {
        return;
      }

      await client.shutdownPool(picked);
      vscode.window.showInformationMessage(`Pool "${picked}" stopped.`);
      onPoolChanged();
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand(
      "skyward.stopPoolInline",
      async (node?: TreeNode) => {
        if (node?.kind !== "pool") {
          vscode.window.showWarningMessage("No pool selected.");
          return;
        }

        await client.shutdownPool(node.name);
        vscode.window.showInformationMessage(`Pool "${node.name}" stopped.`);
        onPoolChanged();
      },
    ),
  );
}
