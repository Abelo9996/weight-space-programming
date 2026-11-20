"""Fit a StatNN-style weight statistics baseline on explicit training IDs."""

import platform

import numpy as np
from torch import nn
import torch

from ..experiment import _source_provenance
from .analysis import accuracy_metrics
from .data import integer, load_panel


QUANTILES = (0, 0.01, 0.25, 0.5, 0.75, 0.99, 1)
HYPERPARAMETERS = {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05, "loss": "squared_error"}


def statistics_features(model):
    """Compute mean, population standard deviation and quantiles by tensor."""
    features = []
    for layer in model:
        if type(layer) not in (nn.Conv2d, nn.Linear):
            continue
        for tensor in (layer.weight, layer.bias):
            values = tensor.detach().cpu().numpy().flatten()
            features.extend([float(values.mean()), float(values.std()), *np.quantile(values, QUANTILES).tolist()])
    return np.array(features, dtype=np.float64)


def run_baseline(panel_path, *, seed=20261119):
    integer(seed, 0, 2**32 - 1, "seed")
    panel = load_panel(panel_path)
    train, test = panel.split("train"), panel.split("test")
    if len(train) < 2:
        raise ValueError("statistics baseline needs at least two distinct explicit training networks")
    try:
        import sklearn
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError as error:
        raise ImportError("install weightspace[reading-baseline] to fit the statistics baseline") from error

    regressor = GradientBoostingRegressor(random_state=seed, **HYPERPARAMETERS)
    regressor.fit(np.stack([statistics_features(row.model) for row in train]), [row.accuracy for row in train])
    predictions = np.clip(regressor.predict(np.stack([statistics_features(row.model) for row in test])), 0, 1)

    metrics = {}
    for activation in ("all", "relu", "tanh"):
        selected = [(row.accuracy, float(value)) for row, value in zip(test, predictions, strict=True)
                    if activation == "all" or row.activation == activation]
        if selected:
            metrics[activation] = accuracy_metrics([row[0] for row in selected], [row[1] for row in selected])
    return {
        "schema_version": 1, "method": "StatNN-style tensor statistics and gradient-boosted trees",
        "panel": panel.provenance(), "package": _source_provenance(),
        "runtime": {
            "python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__),
            "scikit_learn": sklearn.__version__, "platform": platform.platform(),
        },
        "quantiles": list(QUANTILES), "hyperparameters": dict(HYPERPARAMETERS, random_state=seed),
        "training_network_sha256": [row.network_sha256 for row in train],
        "test_predictions": [
            {"id": row.id, "network_sha256": row.network_sha256, "accuracy": row.accuracy, "prediction": float(value)}
            for row, value in zip(test, predictions, strict=True)
        ],
        "metrics": metrics,
    }
