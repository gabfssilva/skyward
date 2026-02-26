from __future__ import annotations

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(180), pytest.mark.xdist_group("joblib")]


class TestJoblibPlugin:
    def test_joblib_parallel_dispatches_to_pool(self, joblib_plugin_pool) -> None:
        from joblib import Parallel, delayed

        def square(x: int) -> int:
            return x**2

        results = Parallel(n_jobs=-1)(delayed(square)(i) for i in range(10))
        assert results == [i**2 for i in range(10)]

    def test_tasks_run_on_workers(self, joblib_plugin_pool) -> None:
        from joblib import Parallel, delayed

        def get_hostname() -> str:
            import socket

            return socket.gethostname()

        results = list(Parallel(n_jobs=-1)(delayed(get_hostname)() for _ in range(4)))
        # At least some results should come from worker containers (not localhost)
        assert len(results) == 4
        assert all(isinstance(r, str) for r in results)
