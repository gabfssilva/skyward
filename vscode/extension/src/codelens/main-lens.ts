/**
 * CodeLens provider that detects `@sky.main` decorators in Python files
 * and offers a "Run on {pool}" action.
 *
 * The lens parses the decorated function's parameter signature so the
 * run command can prompt the user for each argument via input boxes.
 */

import * as vscode from "vscode";

// ── Types ──────────────────────────────────────────────────────

export interface ParsedParam {
  name: string;
  type: string;
  default?: string;
}

// ── Regex ──────────────────────────────────────────────────────

const DECORATOR_RE = /^@sky\.main\s*(\(\))?\s*$/;
const DEF_RE = /^def\s+(\w+)\s*\(([^)]*)\)/;

// ── Provider ───────────────────────────────────────────────────

export class SkyMainCodeLensProvider implements vscode.CodeLensProvider {
  private readonly _onDidChange = new vscode.EventEmitter<void>();
  readonly onDidChangeCodeLenses: vscode.Event<void> = this._onDidChange.event;

  private _activePool = "train";

  setActivePool(name: string): void {
    this._activePool = name;
    this._onDidChange.fire();
  }

  provideCodeLenses(
    document: vscode.TextDocument,
  ): vscode.CodeLens[] {
    const lenses: vscode.CodeLens[] = [];
    const lineCount = document.lineCount;

    for (let i = 0; i < lineCount; i++) {
      const lineText = document.lineAt(i).text.trim();
      if (!DECORATOR_RE.test(lineText)) {
        continue;
      }

      const { fnName, params } = this._parseFunctionDef(document, i + 1);
      if (!fnName) {
        continue;
      }

      const range = document.lineAt(i).range;
      lenses.push(
        new vscode.CodeLens(range, {
          title: `Run on ${this._activePool} \u25b6`,
          command: "skyward.runMain",
          arguments: [document.uri, fnName, params],
        }),
      );
    }

    return lenses;
  }

  // ── Helpers ────────────────────────────────────────────────────

  /**
   * Starting from the line after the decorator, scan up to 2 non-empty
   * lines looking for a `def funcname(params)` pattern.
   */
  private _parseFunctionDef(
    document: vscode.TextDocument,
    startLine: number,
  ): { fnName: string | undefined; params: ParsedParam[] } {
    let checked = 0;

    for (let i = startLine; i < document.lineCount && checked < 2; i++) {
      const text = document.lineAt(i).text.trim();
      if (text === "") {
        continue;
      }
      checked++;

      const match = DEF_RE.exec(text);
      if (match) {
        return {
          fnName: match[1],
          params: this._parseParams(match[2]),
        };
      }
    }

    return { fnName: undefined, params: [] };
  }

  /**
   * Parse a raw parameter string like `epochs: int = 50, lr: float = 0.001`
   * into structured {@link ParsedParam} objects.
   */
  private _parseParams(raw: string): ParsedParam[] {
    if (!raw.trim()) {
      return [];
    }

    return raw.split(",").reduce<ParsedParam[]>((acc, segment) => {
      const trimmed = segment.trim();
      if (!trimmed || trimmed === "self" || trimmed === "cls") {
        return acc;
      }

      const colonIdx = trimmed.indexOf(":");
      const eqIdx = trimmed.indexOf("=");

      let name: string;
      let type = "str";
      let defaultVal: string | undefined;

      if (colonIdx !== -1) {
        name = trimmed.slice(0, colonIdx).trim();
        const afterColon = trimmed.slice(colonIdx + 1).trim();

        if (eqIdx !== -1 && eqIdx > colonIdx) {
          const eqInAfter = afterColon.indexOf("=");
          type = afterColon.slice(0, eqInAfter).trim();
          defaultVal = afterColon.slice(eqInAfter + 1).trim();
        } else {
          type = afterColon;
        }
      } else if (eqIdx !== -1) {
        name = trimmed.slice(0, eqIdx).trim();
        defaultVal = trimmed.slice(eqIdx + 1).trim();
      } else {
        name = trimmed;
      }

      acc.push({ name, type, ...(defaultVal !== undefined && { default: defaultVal }) });
      return acc;
    }, []);
  }
}
