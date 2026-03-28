"""Entry point: python -m skyward.daemon"""

import asyncio
import signal

from skyward.daemon.server import DaemonServer


def main() -> None:
    async def _run() -> None:
        server = DaemonServer()
        async with server:
            server._stop = asyncio.Event()
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, server._stop.set)
            await server._stop.wait()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
