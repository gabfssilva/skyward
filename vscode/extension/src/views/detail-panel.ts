/**
 * WebView panel for pool detail — Overview and Logs tabs.
 *
 * Single panel per pool. Tabs switch via inline JS. Node metrics are
 * shown aggregated in the pool overview, not as separate panels.
 */

import * as vscode from "vscode";
import type { PoolView, SkywardEvent } from "../sidecar/protocol";

export class DetailPanelManager implements vscode.Disposable {
  private _panels = new Map<string, vscode.WebviewPanel>();

  showPoolDetail(view: PoolView): void {
    const key = view.name;
    const panel = this._getOrCreatePanel(key, view.name);
    panel.webview.html = renderHtml(view);
  }

  handleEvent(event: SkywardEvent, getPoolView: (pool: string) => PoolView | undefined): void {
    const panel = this._panels.get(event.pool);
    if (!panel) return;
    const view = getPoolView(event.pool);
    if (view) panel.webview.html = renderHtml(view);
  }

  dispose(): void {
    for (const p of this._panels.values()) p.dispose();
    this._panels.clear();
  }

  private _getOrCreatePanel(key: string, title: string): vscode.WebviewPanel {
    const existing = this._panels.get(key);
    if (existing) { existing.reveal(vscode.ViewColumn.One); return existing; }

    const panel = vscode.window.createWebviewPanel(
      "skyward.pool", title, vscode.ViewColumn.One,
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

function clusterAvg(view: PoolView, metric: string): number | undefined {
  const vals: number[] = [];
  for (const n of Object.values(view.nodes)) {
    if (n.status === "ready" && n.metrics[metric] !== undefined) vals.push(n.metrics[metric]);
  }
  return vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : undefined;
}

function gauge(label: string, value: number, unit: string): string {
  const pct = Math.min(value / 100, 1);
  const deg = pct * 360;
  const color = pct > 0.85 ? "var(--red)" : pct > 0.6 ? "var(--yellow)" : "var(--green)";
  return `<div class="g">
    <div class="g-ring" style="background:conic-gradient(${color} ${deg}deg, var(--track) ${deg}deg)">
      <div class="g-in"><span class="g-val">${value.toFixed(0)}</span><span class="g-unit">${unit}</span></div>
    </div>
    <div class="g-lbl">${esc(label)}</div>
  </div>`;
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
  --track: color-mix(in srgb, var(--fg) 10%, transparent);
  --surface: color-mix(in srgb, var(--fg) 5%, var(--bg));
  --green: var(--vscode-charts-green);
  --yellow: var(--vscode-charts-yellow);
  --red: var(--vscode-charts-red);
  --blue: var(--vscode-charts-blue);
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: var(--vscode-font-family); color: var(--fg); background: var(--bg); line-height: 1.5; }

/* ── Header ─────────────────────────── */
.hdr { padding: 20px 28px 0; }
.hdr h1 { font-size: 1.4em; font-weight: 700; display: inline; margin-right: 8px; }
.badge {
  font-size: 0.6em; font-weight: 700; padding: 2px 9px; border-radius: 9px;
  text-transform: uppercase; letter-spacing: 0.06em; vertical-align: middle;
}
.badge.ready { background: var(--green); color: #000; }
.badge.provisioning,.badge.bootstrapping,.badge.workers,.badge.ssh { background: var(--yellow); color: #000; }
.badge.stopping { background: var(--red); color: #fff; }
.sub { color: var(--dim); font-size: 0.82em; margin-top: 3px; }

/* ── Tabs ───────────────────────────── */
.tabs { display: flex; gap: 0; padding: 16px 28px 0; border-bottom: 1px solid var(--border); }
.tab {
  padding: 8px 20px; font-size: 0.8em; font-weight: 600; cursor: pointer;
  color: var(--dim); border-bottom: 2px solid transparent; margin-bottom: -1px;
  text-transform: uppercase; letter-spacing: 0.06em; background: none; border-top: none; border-left: none; border-right: none;
}
.tab:hover { color: var(--fg); }
.tab.active { color: var(--fg); border-bottom-color: var(--blue); }
.pane { display: none; padding: 20px 28px 28px; }
.pane.active { display: block; }

/* ── Section ────────────────────────── */
.sec { margin-bottom: 24px; }
.sec-t {
  font-size: 0.68em; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.1em; color: var(--dim); margin-bottom: 10px; padding-bottom: 5px;
  border-bottom: 1px solid var(--border);
}

/* ── Gauges ──────────────────────────── */
.gauges { display: flex; gap: 28px; flex-wrap: wrap; }
.g { text-align: center; }
.g-ring {
  width: 84px; height: 84px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center; margin: 0 auto 5px;
}
.g-in {
  width: 62px; height: 62px; border-radius: 50%; background: var(--bg);
  display: flex; align-items: center; justify-content: center; flex-direction: column;
}
.g-val { font-size: 1.1em; font-weight: 700; line-height: 1; }
.g-unit { font-size: 0.58em; color: var(--dim); }
.g-lbl { font-size: 0.68em; font-weight: 600; color: var(--dim); text-transform: uppercase; letter-spacing: 0.05em; }

/* ── Cards ───────────────────────────── */
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 8px; }
.c { background: var(--surface); border: 1px solid var(--border); border-radius: 7px; padding: 10px 12px; }
.c-l { font-size: 0.62em; font-weight: 700; color: var(--dim); text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 3px; }
.c-v { font-size: 1.3em; font-weight: 700; line-height: 1.2; }
.c-v span { font-size: 0.45em; font-weight: 400; color: var(--dim); margin-left: 2px; }
.green { color: var(--green); } .yellow { color: var(--yellow); }
.red { color: var(--red); } .blue { color: var(--blue); } .dim { color: var(--dim); }

/* ── Table ───────────────────────────── */
table { width: 100%; border-collapse: collapse; }
th {
  font-size: 0.62em; font-weight: 700; color: var(--dim); text-transform: uppercase;
  letter-spacing: 0.07em; text-align: left; padding: 6px 12px 6px 0;
  border-bottom: 2px solid var(--border);
}
th.r { text-align: right; }
td { padding: 7px 12px 7px 0; border-bottom: 1px solid var(--border); font-size: 0.82em; }
td.r { text-align: right; font-variant-numeric: tabular-nums; }
td.mono { font-family: var(--vscode-editor-font-family); }
tr:last-child td { border-bottom: none; }
.dot {
  display: inline-block; width: 7px; height: 7px; border-radius: 50%;
  margin-right: 5px; vertical-align: middle;
}
.dot.ready { background: var(--green); }
.dot.bootstrapping { background: var(--yellow); }
.dot.waiting,.dot.ssh { background: var(--dim); }
col.narrow { width: 60px; }
col.status { width: 110px; }
col.accel { width: 110px; }
col.ip { width: 110px; }

/* ── Log ─────────────────────────────── */
.log-list { font-family: var(--vscode-editor-font-family); font-size: 0.8em; line-height: 1.7; }
.log-row { display: flex; gap: 10px; padding: 2px 0; border-bottom: 1px solid color-mix(in srgb, var(--fg) 3%, transparent); }
.log-row:last-child { border-bottom: none; }
.log-ts { color: var(--dim); flex-shrink: 0; width: 60px; text-align: right; }
.log-lvl { flex-shrink: 0; width: 40px; font-weight: 600; text-transform: uppercase; font-size: 0.9em; }
.log-lvl.info { color: var(--blue); }
.log-lvl.warn { color: var(--yellow); }
.log-lvl.error { color: var(--red); }
.log-node { color: var(--dim); flex-shrink: 0; width: 52px; }
.log-msg { flex: 1; word-break: break-word; }

/* ── Pagination ──────────────────────── */
.page-ctl {
  display: flex; align-items: center; gap: 8px; margin-top: 8px;
  font-size: 0.75em; color: var(--dim);
}
.page-btn {
  background: var(--surface); border: 1px solid var(--border); border-radius: 4px;
  padding: 3px 10px; font-size: 0.85em; color: var(--fg); cursor: pointer;
}
.page-btn:hover { border-color: var(--blue); }
.page-btn:disabled { opacity: 0.3; cursor: default; }
`;

// ── JS (inline, minimal) ────────────────────────────────────────

const JS = `
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.pane').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(tab.dataset.pane).classList.add('active');
  });
});
document.querySelectorAll('.page-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const table = btn.closest('.sec').querySelector('table');
    const rows = table.querySelectorAll('tbody tr');
    const size = 10;
    const page = parseInt(btn.dataset.page);
    rows.forEach((r, i) => r.style.display = (i >= page * size && i < (page + 1) * size) ? '' : 'none');
    btn.closest('.page-ctl').querySelector('.page-info').textContent =
      'Page ' + (page + 1) + ' of ' + Math.ceil(rows.length / size);
    btn.closest('.page-ctl').querySelectorAll('.page-btn').forEach(b => b.disabled = false);
    if (page === 0) btn.closest('.page-ctl').querySelector('[data-dir=prev]').disabled = true;
    if ((page + 1) * size >= rows.length) btn.closest('.page-ctl').querySelector('[data-dir=next]').disabled = true;
    btn.closest('.page-ctl').dataset.page = page;
  });
});
`;

// ── Render ───────────────────────────────────────────────────────

function renderHtml(view: PoolView): string {
  const ready = Object.values(view.nodes).filter((n) => n.status === "ready").length;
  const t = view.tasks;
  const s = view.scaling;

  let h = `<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<style>${CSS}</style></head><body>`;

  // ── Header
  h += `<div class="hdr">
    <h1>${esc(view.name)}</h1><span class="badge ${view.phase}">${esc(view.phase)}</span>
    <div class="sub">${ready}/${view.total_nodes} nodes · $${view.cost_per_hour.toFixed(2)}/hr · $${view.cost_total.toFixed(2)} spent</div>
  </div>`;

  h += `<div style="padding: 20px 28px 28px">`;

  // Cluster utilization
  const gpuAvg = clusterAvg(view, "gpu_util");
  const vramAvg = clusterAvg(view, "vram");
  const cpuAvg = clusterAvg(view, "cpu");
  const memAvg = clusterAvg(view, "mem");
  if (gpuAvg !== undefined || cpuAvg !== undefined) {
    h += `<div class="sec"><div class="sec-t">Cluster Utilization</div><div class="gauges">`;
    if (gpuAvg !== undefined) h += gauge("GPU", gpuAvg, "%");
    if (vramAvg !== undefined) h += gauge("VRAM", vramAvg, "%");
    if (cpuAvg !== undefined) h += gauge("CPU", cpuAvg, "%");
    if (memAvg !== undefined) h += gauge("Memory", memAvg, "%");
    h += `</div></div>`;
  }

  // Cost
  h += `<div class="sec"><div class="sec-t">Cost</div><div class="cards">
    <div class="c"><div class="c-l">Hourly Rate</div><div class="c-v">$${view.cost_per_hour.toFixed(2)}<span>/hr</span></div></div>
    <div class="c"><div class="c-l">Spent</div><div class="c-v">$${view.cost_total.toFixed(2)}</div></div>
  </div></div>`;

  // Tasks
  h += `<div class="sec"><div class="sec-t">Tasks</div><div class="cards">
    <div class="c"><div class="c-l">Completed</div><div class="c-v green">${t.done}</div></div>
    <div class="c"><div class="c-l">Running</div><div class="c-v blue">${t.running}</div></div>
    <div class="c"><div class="c-l">Queued</div><div class="c-v">${t.queued}</div></div>
    <div class="c"><div class="c-l">Failed</div><div class="c-v${t.failed > 0 ? " red" : ""}">${t.failed}</div></div>
    <div class="c"><div class="c-l">Throughput</div><div class="c-v blue">${t.throughput.toFixed(1)}<span>/s</span></div></div>
    <div class="c"><div class="c-l">Avg Latency</div><div class="c-v">${t.avg_latency > 0 ? t.avg_latency.toFixed(2) : "—"}<span>${t.avg_latency > 0 ? "s" : ""}</span></div></div>
  </div></div>`;

  // Functions
  const fns = Object.entries(t.fn_summary);
  if (fns.length > 0) {
    h += `<div class="sec"><div class="sec-t">Functions</div><table>
      <thead><tr><th>Function</th><th class="r">Calls</th><th class="r">Avg</th><th class="r">Min</th><th class="r">Max</th><th class="r">Failed</th></tr></thead><tbody>`;
    // th.r already aligns headers right to match td.r values
    for (const [name, s] of fns) {
      h += `<tr><td><strong>${esc(name)}</strong></td>
        <td class="r">${s.calls}</td><td class="r mono">${s.avg.toFixed(2)}s</td>
        <td class="r mono">${s.min.toFixed(2)}s</td><td class="r mono">${s.max.toFixed(2)}s</td>
        <td class="r${s.failed > 0 ? " red" : ""}">${s.failed}</td></tr>`;
    }
    h += `</tbody></table></div>`;
  }

  // Nodes with per-node metrics
  const nodes = Object.values(view.nodes);
  if (nodes.length > 0) {
    const pageSize = 10;
    const totalPages = Math.ceil(nodes.length / pageSize);

    h += `<div class="sec"><div class="sec-t">Nodes (${nodes.length})</div><table>
      <colgroup>
        <col style="width:70px"><col class="status"><col class="accel"><col class="ip">
        <col class="narrow"><col class="narrow"><col class="narrow"><col class="narrow"><col class="narrow">
      </colgroup>
      <thead><tr><th>ID</th><th>Status</th><th>Accelerator</th><th>IP</th><th class="r">GPU</th><th class="r">VRAM</th><th class="r">CPU</th><th class="r">Mem</th><th class="r">Tasks</th></tr></thead><tbody>`;
    for (let i = 0; i < nodes.length; i++) {
      const n = nodes[i];
      const tasks = t.tasks_per_node[n.node_id] ?? 0;
      const display = i < pageSize ? "" : ' style="display:none"';
      const m = (k: string) => n.metrics[k] !== undefined ? `${n.metrics[k].toFixed(0)}%` : "—";
      h += `<tr${display}>
        <td>node-${n.node_id}</td>
        <td><span class="dot ${n.status}"></span>${esc(n.status)}</td>
        <td>${esc(n.accelerator ?? "—")}</td>
        <td class="mono">${esc(n.ip ?? "—")}</td>
        <td class="r mono">${m("gpu_util")}</td>
        <td class="r mono">${m("vram")}</td>
        <td class="r mono">${m("cpu")}</td>
        <td class="r mono">${m("mem")}</td>
        <td class="r">${tasks}</td>
      </tr>`;
    }
    h += `</tbody></table>`;

    if (totalPages > 1) {
      h += `<div class="page-ctl" data-page="0">
        <button class="page-btn" data-dir="prev" data-page="0" disabled>&larr; Prev</button>
        <span class="page-info">Page 1 of ${totalPages}</span>
        <button class="page-btn" data-dir="next" data-page="1">Next &rarr;</button>
      </div>`;
    }
    h += `</div>`;
  }

  // Scaling (only if interesting)
  if (s.is_elastic || s.pending > 0 || s.draining > 0) {
    h += `<div class="sec"><div class="sec-t">Scaling</div><div class="cards">
      <div class="c"><div class="c-l">Desired</div><div class="c-v">${s.desired}</div></div>`;
    if (s.pending > 0) h += `<div class="c"><div class="c-l">Pending</div><div class="c-v yellow">${s.pending}</div></div>`;
    if (s.draining > 0) h += `<div class="c"><div class="c-l">Draining</div><div class="c-v red">${s.draining}</div></div>`;
    if (s.is_elastic && s.min_nodes !== undefined && s.max_nodes !== undefined) {
      h += `<div class="c"><div class="c-l">Range</div><div class="c-v">${s.min_nodes}<span> – </span>${s.max_nodes}</div></div>`;
    }
    h += `</div></div>`;
  }

  h += `</div>`; // end content

  h += `<script>${JS}</script></body></html>`;
  return h;
}
