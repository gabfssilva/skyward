from __future__ import annotations

import pytest

import skyward as sky

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.timeout(300),
    pytest.mark.slow,
]


@pytest.mark.xdist_group("torch")
class TestTorchIntegration:
    def test_torch_sets_env_vars(self, torch_pool):
        @sky.compute
        @sky.integrations.torch(backend="gloo")
        def check_env():
            import os

            return {
                "MASTER_ADDR": os.environ.get("MASTER_ADDR"),
                "WORLD_SIZE": os.environ.get("WORLD_SIZE"),
                "RANK": os.environ.get("RANK"),
            }

        results = check_env() @ torch_pool
        for result in results:
            assert result["MASTER_ADDR"] is not None
            assert result["WORLD_SIZE"] == "2"
            assert result["RANK"] in ("0", "1")

    def test_torch_init_process_group(self, torch_pool):
        @sky.compute
        @sky.integrations.torch(backend="gloo")
        def check_distributed():
            import torch.distributed as dist

            return dist.is_initialized()

        results = check_distributed() @ torch_pool
        assert all(results)


@pytest.mark.xdist_group("torch")
class TestTransformersIntegration:
    def test_transformers_sets_torch_env(self, torch_pool):
        @sky.compute
        @sky.integrations.transformers(backend="gloo")
        def check_env():
            import os

            return {
                "MASTER_ADDR": os.environ.get("MASTER_ADDR"),
                "WORLD_SIZE": os.environ.get("WORLD_SIZE"),
                "RANK": os.environ.get("RANK"),
            }

        results = check_env() @ torch_pool
        for result in results:
            assert result["MASTER_ADDR"] is not None
            assert result["WORLD_SIZE"] == "2"
            assert result["RANK"] in ("0", "1")

    def test_transformers_init_process_group(self, torch_pool):
        @sky.compute
        @sky.integrations.transformers(backend="gloo")
        def check_distributed():
            import torch.distributed as dist

            return dist.is_initialized()

        results = check_distributed() @ torch_pool
        assert all(results)


@pytest.mark.xdist_group("jax")
class TestJaxIntegration:
    def test_jax_sets_env_vars(self, jax_pool):
        @sky.compute
        @sky.integrations.jax()
        def check_env():
            import os

            return {
                "JAX_COORDINATOR_ADDRESS": os.environ.get("JAX_COORDINATOR_ADDRESS"),
                "JAX_NUM_PROCESSES": os.environ.get("JAX_NUM_PROCESSES"),
                "JAX_PROCESS_ID": os.environ.get("JAX_PROCESS_ID"),
            }

        results = check_env() @ jax_pool
        for result in results:
            assert result["JAX_COORDINATOR_ADDRESS"] is not None
            assert result["JAX_NUM_PROCESSES"] == "2"
            assert result["JAX_PROCESS_ID"] in ("0", "1")

    def test_jax_distributed_init(self, jax_pool):
        @sky.compute
        @sky.integrations.jax()
        def check_distributed():
            import jax

            return {
                "initialized": jax.distributed.is_initialized(),
                "process_count": jax.process_count(),
                "process_index": jax.process_index(),
            }

        results = check_distributed() @ jax_pool
        for result in results:
            assert result["initialized"]
            assert result["process_count"] == 2
            assert result["process_index"] in (0, 1)


@pytest.mark.xdist_group("keras")
class TestKerasIntegration:
    def test_keras_torch_backend_env(self, keras_pool):
        @sky.compute
        @sky.integrations.keras(backend="torch")
        def check_env():
            import os

            import torch.distributed as dist

            return {
                "MASTER_ADDR": os.environ.get("MASTER_ADDR"),
                "WORLD_SIZE": os.environ.get("WORLD_SIZE"),
                "RANK": os.environ.get("RANK"),
                "torch_initialized": dist.is_initialized(),
            }

        results = check_env() @ keras_pool
        for result in results:
            assert result["MASTER_ADDR"] is not None
            assert result["WORLD_SIZE"] == "2"
            assert result["RANK"] in ("0", "1")
            assert result["torch_initialized"]

    def test_keras_backend_and_distribution(self, keras_pool):
        @sky.compute
        @sky.integrations.keras(backend="torch")
        def check_keras():
            import keras

            return {
                "backend": keras.backend.backend(),
                "has_distribution": keras.distribution.distribution() is not None,
            }

        results = check_keras() @ keras_pool
        for result in results:
            assert result["backend"] == "torch"


@pytest.mark.xdist_group("parallel")
class TestJoblibIntegration:
    def test_joblib_parallel_distributes(self, parallel_pool):
        from joblib import Parallel, delayed

        from skyward.integrations.joblib import joblib_backend

        def square(x):
            return x ** 2

        with joblib_backend(parallel_pool):
            results = list(Parallel(n_jobs=-1)(delayed(square)(i) for i in range(20)))  # type: ignore[reportArgumentType]

        assert sorted(results) == [i ** 2 for i in range(20)]  # type: ignore[reportArgumentType]

    def test_joblib_parallel_with_imports(self, parallel_pool):
        from joblib import Parallel, delayed

        from skyward.integrations.joblib import joblib_backend

        def compute_hash(text):
            import hashlib

            return hashlib.md5(text.encode()).hexdigest()

        with joblib_backend(parallel_pool):
            inputs = [f"test_{i}" for i in range(10)]
            results = list(Parallel(n_jobs=-1)(delayed(compute_hash)(t) for t in inputs))  # type: ignore[reportArgumentType]

        assert len(results) == 10
        assert all(isinstance(r, str) and len(r) == 32 for r in results)


@pytest.mark.xdist_group("parallel")
class TestScikitLearnIntegration:
    def test_sklearn_grid_search(self, parallel_pool):
        from sklearn.datasets import make_classification
        from sklearn.model_selection import GridSearchCV
        from sklearn.tree import DecisionTreeClassifier

        from skyward.integrations.joblib import sklearn_backend

        x_data, y = make_classification(n_samples=100, n_features=10, random_state=42)

        with sklearn_backend(parallel_pool):
            grid = GridSearchCV(
                DecisionTreeClassifier(),
                {"max_depth": [2, 4, 6, 8]},
                cv=3,
                n_jobs=-1,
            )
            grid.fit(x_data, y)

        assert grid.best_params_ is not None
        assert grid.best_score_ > 0.5
