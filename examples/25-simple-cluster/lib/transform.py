import numpy as np

from .config import ExperimentConfig


def normalize(data: np.ndarray) -> np.ndarray:
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std[std == 0] = 1.0
    return (data - mean) / std


def summarize(data: np.ndarray, config: ExperimentConfig) -> dict:
    return {
        "shape": data.shape,
        "mean": float(data.mean()),
        "std": float(data.std()),
        "seed": config.seed,
    }
