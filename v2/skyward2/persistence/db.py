from pathlib import Path

from piccolo.engine.sqlite import SQLiteEngine

from skyward2.persistence.tables import TABLES

DEFAULT_PATH = Path.home() / ".skyward" / "skyward.sqlite"

PRAGMAS = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA foreign_keys=ON",
    "PRAGMA busy_timeout=5000",
    "PRAGMA synchronous=NORMAL",
)


async def connect(path: Path | None = None) -> SQLiteEngine:
    """Open the store and make it usable by more than one process.

    WAL is what lets a `sky.Compute` in one script and the daemon in another
    write to the same database without an exclusive lock file. It is a journal
    mode, not a change feed — nothing reads the WAL itself.
    """
    target = path or DEFAULT_PATH
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    engine = SQLiteEngine(path=str(target))
    for table in TABLES:
        table._meta.db = engine

    for pragma in PRAGMAS:
        await TABLES[0].raw(pragma).run()

    for table in TABLES:
        await table.create_table(if_not_exists=True).run()

    return engine
