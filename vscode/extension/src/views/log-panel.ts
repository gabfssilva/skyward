/**
 * Dedicated log panel — centralized logs with per-node filtering.
 *
 * Opens a WebView with a filter bar: "All | node-0 | node-1 | ..."
 * Logs are shown newest-first. Filter switches via inline JS.
 */

import * as vscode from "vscode";
import type { LogEntry, PoolView, SkywardEvent } from "../sidecar/protocol";

export class LogPanelManager implements vscode.Disposable {
  private _panels = new Map<string, vscode.WebviewPanel>();

  showLogs(view: PoolView): void {
    const key = view.name;
    const panel = this._getOrCreatePanel(key, `Logs: ${view.name}`);
    panel.webview.html = renderLogHtml(view);
  }

  handleEvent(event: SkywardEvent, getPoolView: (pool: string) => PoolView | undefined): void {
    const panel = this._panels.get(event.pool);
    if (!panel) return;
    const view = getPoolView(event.pool);
    if (view) panel.webview.html = renderLogHtml(view);
  }

  dispose(): void {
    for (const p of this._panels.values()) p.dispose();
    this._panels.clear();
  }

  private _getOrCreatePanel(key: string, title: string): vscode.WebviewPanel {
    const existing = this._panels.get(key);
    if (existing) { existing.reveal(vscode.ViewColumn.One); return existing; }

    const panel = vscode.window.createWebviewPanel(
      "skyward.logs", title, vscode.ViewColumn.One,
      { enableScripts: true, retainContextWhenHidden: true },
    );
    panel.onDidDispose(() => this._panels.delete(key));
    this._panels.set(key, panel);
    return panel;
  }
}

// ── Helpers ──────────────────────────────────────────────────────

function esc(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function timeAgo(ts: number): string {
  const s = Math.floor(Date.now() / 1000 - ts);
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  return `${Math.floor(s / 3600)}h ago`;
}

// ── CSS ──────────────────────────────────────────────────────────

const CSS = `
:root {
  --bg: var(--vscode-editor-background);
  --fg: var(--vscode-foreground);
  --dim: var(--vscode-descriptionForeground);
  --border: color-mix(in srgb, var(--fg) 8%, transparent);
  --surface: color-mix(in srgb, var(--fg) 5%, var(--bg));
  --green: var(--vscode-charts-green);
  --yellow: var(--vscode-charts-yellow);
  --red: var(--vscode-charts-red);
  --blue: var(--vscode-charts-blue);
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: var(--vscode-font-family); color: var(--fg);
  background: var(--bg); line-height: 1.5;
}

/* header */
.hdr { padding: 20px 24px 0; }
.hdr h1 { font-size: 1.3em; font-weight: 700; margin-bottom: 2px; }
.hdr .sub { color: var(--dim); font-size: 0.82em; }

/* filter bar */
.filters {
  display: flex; gap: 0; padding: 14px 24px 0;
  border-bottom: 1px solid var(--border); flex-wrap: wrap;
}
.filter-btn {
  padding: 7px 16px; font-size: 0.75em; font-weight: 600; cursor: pointer;
  color: var(--dim); border: none; border-bottom: 2px solid transparent;
  margin-bottom: -1px; background: none;
  text-transform: uppercase; letter-spacing: 0.05em;
}
.filter-btn:hover { color: var(--fg); }
.filter-btn.active { color: var(--fg); border-bottom-color: var(--blue); }

/* log content */
.log-content { padding: 12px 24px 24px; }
.log-list {
  font-family: var(--vscode-editor-font-family);
  font-size: 0.8em; line-height: 1.8;
}
.log-row {
  display: flex; gap: 0; padding: 3px 0;
  border-bottom: 1px solid color-mix(in srgb, var(--fg) 3%, transparent);
}
.log-row:last-child { border-bottom: none; }
.log-row.hidden { display: none; }

.log-cell { padding: 0 10px 0 0; flex-shrink: 0; }
.log-ts { width: 65px; color: var(--dim); text-align: right; }
.log-lvl { width: 46px; font-weight: 700; text-transform: uppercase; font-size: 0.9em; }
.log-lvl.info { color: var(--blue); }
.log-lvl.warn { color: var(--yellow); }
.log-lvl.error { color: var(--red); }
.log-lvl.debug { color: var(--dim); }
.log-node { width: 60px; color: var(--dim); }
.log-msg { flex: 1; word-break: break-word; }

.empty { color: var(--dim); font-size: 0.85em; padding: 20px 0; }

/* count badge */
.filter-count {
  font-size: 0.85em; font-weight: 400; color: var(--dim);
  margin-left: 4px;
}
`;

// ── JS ──────────────────────────────────────────────────────────

const JS = `
document.querySelectorAll('.filter-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const node = btn.dataset.node;
    document.querySelectorAll('.log-row').forEach(row => {
      if (node === 'all' || row.dataset.node === node) {
        row.classList.remove('hidden');
      } else {
        row.classList.add('hidden');
      }
    });
  });
});
`;

// ── Render ───────────────────────────────────────────────────────

function renderLogHtml(view: PoolView): string {
  const logs = view.logs;
  const nodeIds = [...new Set(logs.map((l) => l.node_id).filter((id) => id !== undefined))].sort(
    (a, b) => (a as number) - (b as number),
  ) as number[];

  // Count logs per node
  const counts = new Map<string, number>();
  counts.set("all", logs.length);
  for (const log of logs) {
    if (log.node_id !== undefined) {
      const key = String(log.node_id);
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
  }

  let h = `<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<style>${CSS}</style></head><body>`;

  // Header
  h += `<div class="hdr">
    <h1>Logs</h1>
    <div class="sub">${esc(view.name)} · ${logs.length} entries</div>
  </div>`;

  // Filter bar
  h += `<div class="filters">
    <button class="filter-btn active" data-node="all">All<span class="filter-count">(${logs.length})</span></button>`;
  for (const id of nodeIds) {
    const c = counts.get(String(id)) ?? 0;
    h += `<button class="filter-btn" data-node="${id}">node-${id}<span class="filter-count">(${c})</span></button>`;
  }
  h += `</div>`;

  // Log list
  h += `<div class="log-content">`;
  if (logs.length === 0) {
    h += `<div class="empty">No logs yet.</div>`;
  } else {
    h += `<div class="log-list">`;
    // Newest first
    for (let i = logs.length - 1; i >= 0; i--) {
      const log = logs[i];
      const lvl = log.level ?? "info";
      const nodeStr = log.node_id !== undefined ? `node-${log.node_id}` : "";
      const nodeData = log.node_id !== undefined ? String(log.node_id) : "";
      h += `<div class="log-row" data-node="${nodeData}">
        <span class="log-cell log-ts">${timeAgo(log.ts)}</span>
        <span class="log-cell log-lvl ${lvl}">${lvl}</span>
        <span class="log-cell log-node">${nodeStr}</span>
        <span class="log-cell log-msg">${esc(log.message)}</span>
      </div>`;
    }
    h += `</div>`;
  }
  h += `</div>`;

  h += `<script>${JS}</script></body></html>`;
  return h;
}
