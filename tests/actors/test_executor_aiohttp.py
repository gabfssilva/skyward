def test_executor_uses_aiohttp():
    """Verify executor module uses aiohttp, not httpx."""
    import ast
    from pathlib import Path

    source = Path("skyward/executor.py").read_text()
    tree = ast.parse(source)

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)

    assert "aiohttp" in imports, "executor should import aiohttp"
    assert "httpx" not in imports, "executor should not import httpx"
