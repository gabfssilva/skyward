/**
 * Pool lifecycle commands: start, stop, add provider, add pool, inline-stop.
 *
 * Registers commands that drive pool lifecycle through the sidecar
 * client and keep the status bar and explorer in sync.
 */

import * as vscode from "vscode";
import type { SidecarClient } from "../sidecar/protocol";
import type { PoolStatusBar } from "../statusbar/pool-status";
import type { PoolExplorer, TreeNode } from "../views/pool-explorer";

// ── Constants ──────────────────────────────────────────────────

const PROVIDER_TYPES = [
  { label: "AWS", value: "aws" },
  { label: "GCP", value: "gcp" },
  { label: "Hyperstack", value: "hyperstack" },
  { label: "JarvisLabs", value: "jarvislabs" },
  { label: "Lambda", value: "lambda" },
  { label: "RunPod", value: "runpod" },
  { label: "Scaleway", value: "scaleway" },
  { label: "TensorDock", value: "tensordock" },
  { label: "VastAI", value: "vastai" },
  { label: "Verda", value: "verda" },
  { label: "Vultr", value: "vultr" },
];

type AccelItem = vscode.QuickPickItem & { value: string | null };

const ACCELERATORS: AccelItem[] = [
  { label: "$(dash) None (CPU only)", value: "", kind: vscode.QuickPickItemKind.Default },
  { label: "Blackwell", kind: vscode.QuickPickItemKind.Separator, value: null },
  { label: "GB300", description: "288 GB", value: "GB300" },
  { label: "GB200", description: "384 GB", value: "GB200" },
  { label: "B300", description: "288 GB", value: "B300" },
  { label: "B200", description: "192 GB", value: "B200" },
  { label: "B100", description: "192 GB", value: "B100" },
  { label: "Hopper", kind: vscode.QuickPickItemKind.Separator, value: null },
  { label: "H200", description: "141 GB", value: "H200" },
  { label: "H200-NVL", description: "141 GB", value: "H200-NVL" },
  { label: "H100", description: "80 GB", value: "H100" },
  { label: "H100-NVL", description: "94 GB", value: "H100-NVL" },
  { label: "GH200", description: "96 GB", value: "GH200" },
  { label: "Ada Lovelace", kind: vscode.QuickPickItemKind.Separator, value: null },
  { label: "L40S", description: "48 GB", value: "L40S" },
  { label: "L40", description: "48 GB", value: "L40" },
  { label: "L4", description: "24 GB", value: "L4" },
  { label: "RTX 4090", description: "24 GB", value: "RTX4090" },
  { label: "Ampere", kind: vscode.QuickPickItemKind.Separator, value: null },
  { label: "A100", description: "80 GB", value: "A100" },
  { label: "A40", description: "48 GB", value: "A40" },
  { label: "A10G", description: "24 GB", value: "A10G" },
  { label: "A10", description: "24 GB", value: "A10" },
  { label: "RTX 3090", description: "24 GB", value: "RTX3090" },
  { label: "RTX A6000", description: "48 GB", value: "RTXA6000" },
  { label: "Volta / Turing", kind: vscode.QuickPickItemKind.Separator, value: null },
  { label: "V100", description: "32 GB", value: "V100" },
  { label: "T4", description: "16 GB", value: "T4" },
  { label: "AMD", kind: vscode.QuickPickItemKind.Separator, value: null },
  { label: "MI300X", description: "192 GB", value: "MI300X" },
  { label: "MI250X", description: "128 GB", value: "MI250X" },
];

// ── TOML helpers ───────────────────────────────────────────────

function getTomlUri(): vscode.Uri | undefined {
  const folders = vscode.workspace.workspaceFolders;
  if (!folders) {
    vscode.window.showWarningMessage("Open a workspace folder first.");
    return undefined;
  }
  return vscode.Uri.joinPath(folders[0].uri, "skyward.toml");
}

async function appendToml(uri: vscode.Uri, snippet: string): Promise<void> {
  try {
    const existing = await vscode.workspace.fs.readFile(uri);
    const content = new TextDecoder().decode(existing);
    const updated = content.trimEnd() + "\n\n" + snippet;
    await vscode.workspace.fs.writeFile(uri, new TextEncoder().encode(updated));
  } catch {
    await vscode.workspace.fs.writeFile(uri, new TextEncoder().encode(snippet));
  }
}

// ── Add Provider wizard ────────────────────────────────────────

async function runAddProviderWizard(): Promise<string | undefined> {
  const providerType = await vscode.window.showQuickPick(PROVIDER_TYPES, {
    placeHolder: "Cloud provider type",
  });
  if (!providerType) return undefined;

  const name = await vscode.window.showInputBox({
    prompt: "Provider name (used as reference in pools)",
    value: providerType.value,
    validateInput: (v) =>
      v && /^[a-z][a-z0-9_-]*$/.test(v) ? null : "Lowercase letters, numbers, hyphens, underscores",
  });
  if (!name) return undefined;

  const tomlUri = getTomlUri();
  if (!tomlUri) return undefined;

  const snippet = `[providers.${name}]\ntype = "${providerType.value}"\n`;
  await appendToml(tomlUri, snippet);

  const doc = await vscode.workspace.openTextDocument(tomlUri);
  await vscode.window.showTextDocument(doc);
  vscode.window.showInformationMessage(
    `Provider "${name}" added. Edit skyward.toml to set region, API keys, etc.`,
  );

  return name;
}

// ── Add Pool wizard ────────────────────────────────────────────

async function runAddPoolWizard(providerNames: string[]): Promise<string | undefined> {
  const name = await vscode.window.showInputBox({
    prompt: "Pool name",
    placeHolder: "train",
    validateInput: (v) =>
      v && /^[a-z][a-z0-9_-]*$/.test(v) ? null : "Lowercase letters, numbers, hyphens, underscores",
  });
  if (!name) return undefined;

  const provider =
    providerNames.length === 1
      ? providerNames[0]
      : await vscode.window.showQuickPick(providerNames, { placeHolder: "Provider" });
  if (!provider) return undefined;

  const accelerator = await vscode.window.showQuickPick<AccelItem>(ACCELERATORS, {
    placeHolder: "GPU accelerator",
  });
  if (accelerator === undefined) return undefined;

  const nodesStr = await vscode.window.showInputBox({
    prompt: "Number of nodes",
    value: "1",
    validateInput: (v) => (/^\d+$/.test(v) && parseInt(v) > 0 ? null : "Must be a positive integer"),
  });
  if (!nodesStr) return undefined;
  const nodes = parseInt(nodesStr);

  let snippet = `[pools.${name}]\nprovider = "${provider}"\n`;
  if (accelerator.value) {
    snippet += `accelerator = "${accelerator.value}"\n`;
  }
  snippet += `nodes = ${nodes}\n`;

  const tomlUri = getTomlUri();
  if (!tomlUri) return undefined;

  await appendToml(tomlUri, snippet);

  const doc = await vscode.workspace.openTextDocument(tomlUri);
  await vscode.window.showTextDocument(doc);

  return name;
}

// ── Commands ───────────────────────────────────────────────────

export function registerLifecycleCommands(
  context: vscode.ExtensionContext,
  client: SidecarClient,
  statusBar: PoolStatusBar,
  poolExplorer: PoolExplorer,
  onPoolChanged: () => void,
): void {
  // ── Add Provider ──────────────────────────────────────────
  context.subscriptions.push(
    vscode.commands.registerCommand("skyward.addProvider", async () => {
      await runAddProviderWizard();
    }),
  );

  // ── Add Pool ──────────────────────────────────────────────
  context.subscriptions.push(
    vscode.commands.registerCommand("skyward.addPool", async () => {
      let providers = await client.configProviders().catch(() => []);

      if (providers.length === 0) {
        const action = await vscode.window.showWarningMessage(
          "No providers configured. Add a provider first.",
          "Add Provider",
        );
        if (action !== "Add Provider") return;
        const created = await runAddProviderWizard();
        if (!created) return;
        providers = await client.configProviders().catch(() => []);
        if (providers.length === 0) return;
      }

      await runAddPoolWizard(providers);
    }),
  );

  // ── Start Pool ────────────────────────────────────────────
  context.subscriptions.push(
    vscode.commands.registerCommand("skyward.startPool", async (poolName?: string) => {
      let picked = poolName;

      if (!picked) {
        let poolNames = await client.configPools().catch(() => []);

        if (poolNames.length === 0) {
          let providers = await client.configProviders().catch(() => []);

          if (providers.length === 0) {
            const created = await runAddProviderWizard();
            if (!created) return;
            providers = await client.configProviders().catch(() => []);
            if (providers.length === 0) return;
          }

          const created = await runAddPoolWizard(providers);
          if (!created) return;
          poolNames = await client.configPools().catch(() => []);
          if (poolNames.length === 0) return;
        }

        picked =
          poolNames.length === 1
            ? poolNames[0]
            : await vscode.window.showQuickPick(poolNames, { placeHolder: "Select a pool to start" });
      }

      if (!picked) return;

      await client.startDaemon();

      // Poll UI until the pool appears, then stop when ensurePool resolves.
      const poll = setInterval(onPoolChanged, 2000);
      client.ensurePool(picked).then(() => {
        clearInterval(poll);
        onPoolChanged();
      }).catch((err) => {
        clearInterval(poll);
        vscode.window.showErrorMessage(`Pool "${picked}" failed: ${err}`);
      });

      poolExplorer.markStarting(picked);
      vscode.window.showInformationMessage(`Pool "${picked}" starting...`);
      statusBar.selectPool(picked);
      onPoolChanged();
    }),
  );

  // ── Start Pool Inline ──────────────────────────────────────
  context.subscriptions.push(
    vscode.commands.registerCommand(
      "skyward.startPoolInline",
      async (node?: TreeNode) => {
        const name = node?.kind === "stopped-pool" ? node.name
          : node?.kind === "pool" ? node.name
          : undefined;

        if (!name) {
          vscode.window.showWarningMessage("No pool selected.");
          return;
        }

        await client.startDaemon();

        const poll = setInterval(onPoolChanged, 2000);
        client.ensurePool(name).then(() => {
          clearInterval(poll);
          onPoolChanged();
        }).catch((err) => {
          clearInterval(poll);
          vscode.window.showErrorMessage(`Pool "${name}" failed: ${err}`);
        });

        poolExplorer.markStarting(name);
        vscode.window.showInformationMessage(`Pool "${name}" starting...`);
        statusBar.selectPool(name);
        onPoolChanged();
      },
    ),
  );

  // ── Stop Pool ─────────────────────────────────────────────
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

      if (!picked) return;

      await client.shutdownPool(picked);
      vscode.window.showInformationMessage(`Pool "${picked}" stopping...`);
      onPoolChanged();

      // Poll until the pool disappears from the daemon.
      const poll = setInterval(async () => {
        const remaining = await client.listPools().catch(() => []);
        onPoolChanged();
        if (!remaining.some((p) => p.name === picked)) {
          clearInterval(poll);
        }
      }, 2000);
    }),
  );

  // ── Stop Pool Inline ──────────────────────────────────────
  context.subscriptions.push(
    vscode.commands.registerCommand(
      "skyward.stopPoolInline",
      async (node?: TreeNode) => {
        if (node?.kind !== "pool") {
          vscode.window.showWarningMessage("No pool selected.");
          return;
        }

        await client.shutdownPool(node.name);
        vscode.window.showInformationMessage(`Pool "${node.name}" stopping...`);
        onPoolChanged();

        const poll = setInterval(async () => {
          const remaining = await client.listPools().catch(() => []);
          onPoolChanged();
          if (!remaining.some((p) => p.name === node.name)) {
            clearInterval(poll);
          }
        }, 2000);
      },
    ),
  );
}
