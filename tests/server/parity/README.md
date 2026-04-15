# Server parity suite

This directory holds tests that mirror `tests/daemon/` one-for-one with
`daemon=True` replaced by `server=True`. Run via:

```bash
task test:server_parity
```

The suite is gated by `@pytest.mark.server_parity` so the default unit
run stays fast. The plan (§J3) requires the parity baseline to match
`task test:daemon` before `skyward/daemon/` can be deleted (§K2).

Until the matching tests are ported, running this suite is a no-op —
the marker registration lives in `pyproject.toml` so `-m server_parity`
works without warnings.
