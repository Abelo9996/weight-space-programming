import json

import numpy as np
import pytest

from weightspace.readers.analysis import _agreement, _network_intervals, kendall_tau_b
from weightspace.readers.baseline import run_baseline, statistics_features
from weightspace.readers.data import load_panel


def test_kendall_tau_b_includes_ties_and_undefined_values():
    assert kendall_tau_b([1, 2, 3], [3, 2, 1]) == -1
    assert kendall_tau_b([0, 0, 1], [0, 1, 1]) == pytest.approx(0.5)
    assert kendall_tau_b([1, 1, 2, 3], [1, 1, 2, 3]) == 1
    assert kendall_tau_b([0, 0], [0, 1]) is None
    assert kendall_tau_b([0], [0]) is None
    assert kendall_tau_b([], []) is None


def test_agreement_includes_the_decimal_threshold():
    assert _agreement([(0.60, 0.65), (0.30, 0.35), (0.60, 0.65001)], 0.05)["agreement"] == 2 / 3


def test_bootstrap_resamples_networks_and_omits_fixture_or_tiny_intervals():
    values = np.arange(10) / 10
    seed, replicates = 509, 200
    result = _network_intervals(values.tolist(), replicates=replicates, seed=seed, fixtures=False)
    draws = np.random.default_rng(seed).integers(10, size=(replicates, 10))
    expected = np.quantile(values[draws].mean(axis=1), [0.025, 0.975])
    assert result["n_networks"] == 10
    assert result["estimate"] == pytest.approx(0.45)
    assert result["ci95"] == pytest.approx(expected)
    for values, fixtures in ((values[:1], False), (values[:9], False), (values, True)):
        assert _network_intervals(values.tolist(), replicates=replicates, seed=seed, fixtures=fixtures)["ci95"] is None


def test_baseline_fit_uses_only_explicit_training_rows(panel_factory, monkeypatch):
    pytest.importorskip("sklearn")
    import sklearn.ensemble
    original = sklearn.ensemble.GradientBoostingRegressor
    fit_calls = []

    class Spy(original):
        def fit(self, features, scores):
            fit_calls.append((features.copy(), list(scores)))
            return super().fit(features, scores)

    monkeypatch.setattr(sklearn.ensemble, "GradientBoostingRegressor", Spy)
    path, manifest = panel_factory(train=8, test=5)
    panel = load_panel(path)
    first = run_baseline(path, seed=843)
    assert [row.network_sha256 for row in panel.split("train")] == first["training_network_sha256"]
    assert {row["network_sha256"] for row in first["test_predictions"]}.isdisjoint(first["training_network_sha256"])
    assert len(first["test_predictions"]) == 5
    assert first["metrics"]["all"]["n_networks"] == 5

    # A held-out label-only change must leave the fitted model and predictions fixed.
    for row in manifest["networks"]:
        if row["split"] == "test":
            row["accuracy"] = 1.0 - row["accuracy"]
    path.write_text(json.dumps(manifest))
    second = run_baseline(path, seed=843)
    assert len(fit_calls) == 2
    np.testing.assert_array_equal(fit_calls[0][0], fit_calls[1][0])
    np.testing.assert_array_equal(
        fit_calls[0][0], np.stack([statistics_features(row.model) for row in panel.split("train")]),
    )
    assert fit_calls[0][1] == fit_calls[1][1] == [row.accuracy for row in panel.split("train")]
    assert [row["prediction"] for row in first["test_predictions"]] == [row["prediction"] for row in second["test_predictions"]]


def test_baseline_requires_training_networks(panel_factory):
    path, _ = panel_factory(train=0, test=2)
    with pytest.raises(ValueError, match="two distinct explicit training networks"):
        run_baseline(path)
