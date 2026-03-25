from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    n_samples: int = 1000
    n_features: int = 10
