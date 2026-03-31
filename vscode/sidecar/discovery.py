"""AST-based discovery of @sky.main decorated functions."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


def discover_main_functions(files: list[str] | None = None) -> list[dict[str, Any]]:
    """Walk Python files and return metadata for every @sky.main function.

    Parameters
    ----------
    files
        Explicit file paths to scan. When *None*, recursively globs ``*.py``
        from the current working directory.

    Returns
    -------
    list[dict[str, Any]]
        Each dict contains ``name``, ``file``, ``line``, and ``params``.
    """
    if files is None:
        files = [str(p) for p in Path.cwd().rglob("*.py")]

    results: list[dict[str, Any]] = []
    for path in files:
        try:
            source = Path(path).read_text()
            tree = ast.parse(source, filename=path)
        except (SyntaxError, OSError):
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if not _has_sky_main_decorator(node):
                continue

            params = _extract_params(node)
            results.append({
                "name": node.name,
                "file": str(path),
                "line": node.lineno,
                "params": params,
            })

    return results


def _has_sky_main_decorator(node: ast.FunctionDef) -> bool:
    for dec in node.decorator_list:
        match dec:
            case ast.Attribute(value=ast.Name(id="sky"), attr="main"):
                return True
            case ast.Call(func=ast.Attribute(value=ast.Name(id="sky"), attr="main")):
                return True
    return False


def _extract_params(node: ast.FunctionDef) -> list[dict[str, Any]]:
    params: list[dict[str, Any]] = []
    args = node.args

    n_no_default = len(args.args) - len(args.defaults)

    for i, arg in enumerate(args.args):
        if arg.arg in ("self", "cls"):
            continue

        param: dict[str, Any] = {"name": arg.arg}

        if arg.annotation:
            param["type"] = _annotation_to_str(arg.annotation)
        else:
            param["type"] = "str"

        default_idx = i - n_no_default
        if 0 <= default_idx < len(args.defaults):
            param["default"] = _default_to_str(args.defaults[default_idx])

        params.append(param)

    return params


def _annotation_to_str(node: ast.expr) -> str:
    match node:
        case ast.Name(id=name):
            return name
        case ast.Constant(value=val):
            return str(val)
        case _:
            return ast.unparse(node)


def _default_to_str(node: ast.expr) -> str:
    match node:
        case ast.Constant(value=val):
            return repr(val) if isinstance(val, str) else str(val)
        case _:
            return ast.unparse(node)
