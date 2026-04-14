"""SQLite DDL for the persistence layer.

The statements below are copied verbatim from the HTTP server design
(``docs/plans/2026-04-13-http-server-design.md`` §5.2), with ``IF NOT EXISTS``
guards added so :func:`skyward.server.host.migrate.apply_schema` is idempotent.
"""

type DDL = str

SCHEMA: tuple[DDL, ...] = (
    """
    CREATE TABLE IF NOT EXISTS providers (
        name         TEXT PRIMARY KEY,
        type         TEXT NOT NULL,
        config       TEXT NOT NULL CHECK(json_valid(config)),
        created_at   REAL NOT NULL,
        updated_at   REAL NOT NULL,
        last_used_at REAL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_providers_type ON providers(type)",
    """
    CREATE TABLE IF NOT EXISTS compute (
        name                TEXT PRIMARY KEY,
        spec                TEXT NOT NULL CHECK(json_valid(spec)),
        created_at          REAL NOT NULL,

        status_tag          TEXT NOT NULL,
        started_at          REAL,
        stopped_at          REAL,
        stopping_since      REAL,
        failed_at           REAL,
        failure_reason      TEXT,
        nodes_ready         INTEGER,
        last_activity_at    REAL,
        chosen_spec_ordinal INTEGER,
        chosen_spec_json    TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_compute_status   ON compute(status_tag, last_activity_at)",
    "CREATE INDEX IF NOT EXISTS idx_compute_activity ON compute(last_activity_at) WHERE status_tag='ready'",
    """
    CREATE TABLE IF NOT EXISTS nodes (
        id            TEXT PRIMARY KEY,
        compute       TEXT NOT NULL REFERENCES compute(name) ON DELETE CASCADE,
        instance_id   TEXT NOT NULL,
        provider_name TEXT NOT NULL,
        head_addr     TEXT,
        created_at    REAL NOT NULL,

        status_tag    TEXT NOT NULL,
        status_since  REAL,
        bootstrap_phase TEXT,
        lost_at       REAL,
        lost_reason   TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_nodes_compute ON nodes(compute, status_tag)",
    """
    CREATE TABLE IF NOT EXISTS blobs (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        path       TEXT    NOT NULL UNIQUE,
        size       INTEGER NOT NULL,
        sha256     TEXT,
        kind       TEXT    NOT NULL,
        created_at REAL    NOT NULL,
        evicted_at REAL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_blobs_live ON blobs(kind, created_at) WHERE evicted_at IS NULL",
    """
    CREATE TABLE IF NOT EXISTS errors (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        type       TEXT NOT NULL,
        message    TEXT NOT NULL,
        traceback  TEXT,
        created_at REAL NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_errors_type ON errors(type)",
    """
    CREATE TABLE IF NOT EXISTS tasks (
        module   TEXT NOT NULL,
        qualname TEXT NOT NULL,
        PRIMARY KEY (module, qualname)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS task_executions (
        id             TEXT PRIMARY KEY,
        task_module    TEXT NOT NULL,
        task_qualname  TEXT NOT NULL,
        compute        TEXT NOT NULL REFERENCES compute(name),
        kind_tag       TEXT NOT NULL,
        group_id       TEXT,
        payload_blob   INTEGER NOT NULL REFERENCES blobs(id),
        timeout_s      REAL,
        client_id      TEXT,
        submitted_at   REAL NOT NULL,

        status_tag       TEXT NOT NULL,
        finished_at      REAL,
        interrupted_at   REAL,
        interrupt_reason TEXT,
        cancelled_at     REAL,
        cancel_reason    TEXT,

        FOREIGN KEY (task_module, task_qualname) REFERENCES tasks(module, qualname)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_exec_compute_status ON task_executions(compute, status_tag, submitted_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_exec_task           ON task_executions(task_module, task_qualname, submitted_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_exec_group          ON task_executions(group_id) WHERE group_id IS NOT NULL",
    """
    CREATE INDEX IF NOT EXISTS idx_exec_active ON task_executions(status_tag)
        WHERE status_tag IN ('queued','dispatching','running')
    """,
    """
    CREATE TABLE IF NOT EXISTS task_results (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        execution       TEXT NOT NULL REFERENCES task_executions(id) ON DELETE CASCADE,
        shard           INTEGER NOT NULL,
        node_id         TEXT,

        status_tag      TEXT NOT NULL,
        dispatched_at   REAL,
        started_at      REAL,
        finished_at     REAL,
        interrupted_at  REAL,
        interrupt_reason TEXT,
        result_blob     INTEGER REFERENCES blobs(id),
        error_id        INTEGER REFERENCES errors(id),

        UNIQUE (execution, shard)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_results_execution ON task_results(execution)",
    "CREATE INDEX IF NOT EXISTS idx_results_node      ON task_results(node_id, status_tag)",
    """
    CREATE TABLE IF NOT EXISTS events (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        ts         REAL NOT NULL,
        aggregate  TEXT NOT NULL,
        type       TEXT NOT NULL,
        payload    TEXT NOT NULL CHECK(json_valid(payload))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_events_agg ON events(aggregate, id)",
)

__all__ = ["DDL", "SCHEMA"]
