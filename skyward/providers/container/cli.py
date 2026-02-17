from __future__ import annotations

import asyncio
import json


async def run(binary: str, *args: str) -> str:
    proc = await asyncio.create_subprocess_exec(
        binary, *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        cmd = f"{binary} {' '.join(args)}"
        raise RuntimeError(f"{cmd} failed (exit {proc.returncode}): {stderr.decode().strip()}")
    return stdout.decode().strip()


async def run_json(binary: str, *args: str) -> dict | list:
    out = await run(binary, *args)
    return json.loads(out)
