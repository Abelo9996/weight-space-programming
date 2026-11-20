from hashlib import sha256
import json

import numpy as np
import pytest
import torch

from weightspace.readers.data import _build_model


ARCHITECTURE = {
    "input_shape": [1, 4, 4], "convolutions": [{"out_channels": 4, "kernel_size": 2, "stride": 1}],
    "classes": 3,
}


@pytest.fixture
def panel_factory(tmp_path):
    def make(train=0, test=2):
        path = tmp_path / "panel.json"
        manifest = {
            "schema_version": 1,
            "dataset": {"name": "private marker dataset", "source": "https://example.invalid/private-marker", "sampling": "panel"},
            "architecture": ARCHITECTURE, "networks": [],
        }
        for i in range(train + test):
            split = "train" if i < train else "test"
            activation = ("relu", "tanh")[i % 2]
            file = tmp_path / f"private-checkpoint-{i}.npz"
            model = _build_model(ARCHITECTURE, activation)
            generator = torch.Generator().manual_seed(5894 + i)
            state = {
                name: torch.randn(value.shape, generator=generator, dtype=torch.float64).numpy() / 2
                for name, value in model.state_dict().items()
            }
            np.savez(file, **state)
            accuracy = 0.45 + i / 613 if split == "train" else 0.917208631067325 + (i - train) / 800
            manifest["networks"].append({
                "id": f"private-network-{i}", "training_run_id": f"private-run-{i}", "split": split,
                "accuracy": accuracy, "activation": activation,
                "checkpoint": {
                    "format": "pytorch-npz", "path": file.name, "sha256": sha256(file.read_bytes()).hexdigest(),
                    "layout": "state_dict_oihw_oi", "source_checkpoint": f"private-checkpoint-{i}",
                },
            })
        path.write_text(json.dumps(manifest))
        return path, manifest

    return make


@pytest.fixture
def config_factory(tmp_path):
    def make(*, providers=("openai", "anthropic"), ks=(0,), repeats=2, budget="5", credits=None, context=1_000_000):
        raw = {
            "schema_version": 1, "seed": 1142, "repeats": repeats, "in_context": list(ks),
            "agreement_tolerance": 0.05, "budget_usd": budget, "timeout_seconds": 10,
            "confirmed_credits_usd": {provider: "5" for provider in providers} if credits is None else credits,
            "readers": [{
                "provider": provider, "model": f"test-model-{provider}", "effort": "low", "context_window": context,
                "max_output_tokens": 128, "input_tokens_per_byte": 0.4, "framing_tokens": 1024,
                "price": {
                    "source": "https://example.invalid", "as_of": "2026-01-01",
                    "standard": {
                        "input": "1", "cache_read": "0.1", "cache_write": "1.25", "cache_write_long": "2", "output": "5",
                    },
                },
            } for provider in providers],
        }
        path = tmp_path / "config.json"
        path.write_text(json.dumps(raw))
        return path

    return make
