from hashlib import sha256
import json
import os
from pathlib import Path
from shutil import copyfile

import numpy as np
import pytest
import torch

from weightspace.cli import main
from weightspace.readers.data import load_panel, tensor_digest
from weightspace.readers.experiment import load_config, prepare_plan
from weightspace.readers.prompts import EQUIVALENT, variants_for


def test_variants_preserve_logits_and_weights_round_trip(panel_factory):
    path, _ = panel_factory()
    panel = load_panel(path)
    torch_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for row in panel.checkpoints:
            before = tensor_digest(row.model)
            first = variants_for(row, panel.architecture, 4123)
            assert first == variants_for(row, panel.architecture, 4123)
            assert before == tensor_digest(row.model)
            assert len({item.provenance["tensors_sha256"] for item in first.values()}) == 5
            for name in EQUIVALENT:
                assert first[name].provenance["logit_check"]["max_abs_difference"] < 1e-5
            assert first["shuffle"].provenance["logit_check"]["max_abs_difference"] > 1e-5
            assert {item.provenance["network_sha256"] for item in first.values()} == {row.network_sha256}
            original = json.loads(first["original"].text)["parameters"]
            for record, parameter in zip(original, row.model.state_dict().values(), strict=True):
                restored = np.array(record["values"]).reshape(record["shape"])
                np.testing.assert_array_equal(restored, parameter.detach().numpy())
    finally:
        torch.set_num_threads(torch_threads)


@pytest.mark.parametrize("kind", ["source run", "checkpoint", "alias", "tensors"])
def test_train_test_overlap_is_rejected(panel_factory, kind):
    path, manifest = panel_factory(train=1, test=1)
    train, test = manifest["networks"]
    if kind == "source run":
        test["training_run_id"] = "  PRIVATE-RUN-0  "
    elif kind == "checkpoint":
        test["checkpoint"]["source_checkpoint"] = train["checkpoint"]["source_checkpoint"]
    elif kind == "alias":
        test["id"] = train["id"].upper()
    else:
        first = path.parent / train["checkpoint"]["path"]
        converted = path.parent / "rewrapped.npz"
        with np.load(first, allow_pickle=False) as archive:
            np.savez_compressed(converted, **{name: archive[name].astype(">f8") for name in archive.files})
        test["checkpoint"].update(path=converted.name, sha256=sha256(converted.read_bytes()).hexdigest())
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="duplicate"):
        load_panel(path)


def test_rejects_wrong_checkpoint_hash_and_shape(panel_factory):
    path, manifest = panel_factory(test=1)
    row = manifest["networks"][0]
    file = path.parent / row["checkpoint"]["path"]
    file.write_bytes(b"this is not a real checkpoint")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_panel(path)
    np.savez(file, weights=np.zeros(2))
    row["checkpoint"]["sha256"] = sha256(file.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="state_dict"):
        load_panel(path)


def test_generated_equivalent_cannot_overlap_train_checkpoint(panel_factory, config_factory):
    path, manifest = panel_factory(train=1, test=1)
    config = load_config(config_factory())
    plan = prepare_plan(config, path)
    variant = next(iter(plan.variants.values()))["permutation"]
    checkpoint = manifest["networks"][0]["checkpoint"]
    file = path.parent / checkpoint["path"]
    tensors = {
        param["parameter"]: np.asarray(param["values"]).reshape(param["shape"])
        for param in json.loads(variant.text)["parameters"]
    }
    np.savez(file, **tensors)
    checkpoint["sha256"] = sha256(file.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="generated variant overlaps"):
        prepare_plan(config, path)


def test_in_context_prompts_are_train_only_and_paired(panel_factory, config_factory):
    path, manifest = panel_factory(train=32, test=1)
    config = load_config(config_factory(ks=(0, 8, 32)))
    plan = prepare_plan(config, path)
    assert len(plan.metadata["requests"]) == 60
    train_networks = {row.network_sha256 for row in load_panel(path).split("train")}
    for k in (0, 8, 32):
        examples = plan.metadata["examples"][str(k)]
        assert len(examples) == k
        assert set(examples) <= train_networks
        assert plan.metadata["examples"]["32"][:k] == examples
    tests = manifest["networks"][-1]
    matched = {}
    for call in plan.metadata["requests"]:
        prompt = plan.prompt(call)
        assert prompt.count("Labeled example network") == call["k"]
        assert prompt.count('"accuracy":') == call["k"] + 1  # Includes the output instruction.
        for prohibited in ("private-", "private marker", str(tests["accuracy"]), str(path), '"split"'):
            assert prohibited not in prompt
        cell = call["network_sha256"], call["k"], call["condition"]
        assert matched.setdefault(cell, call["prompt_sha256"]) == call["prompt_sha256"]
        assert sha256(prompt.encode()).hexdigest() == call["prompt_sha256"]
    assert len(matched) == 15
    for record in plan.metadata["panel"]["networks"]:
        assert "path" not in record["checkpoint"]
    tests["accuracy"] = 0.6414723825
    path.write_text(json.dumps(manifest))
    relabeled = prepare_plan(config, path)
    assert relabeled.metadata["requests"] == plan.metadata["requests"]


def test_insufficient_training_examples_are_rejected(panel_factory, config_factory):
    path, _ = panel_factory(train=7, test=1)
    with pytest.raises(ValueError, match="k=8 requires.*found 7"):
        prepare_plan(load_config(config_factory(ks=(0, 8))), path)


@pytest.mark.skipif(not os.getenv("WEIGHTSPACE_CNN_ZOO_FIXTURES"), reason="optional archived CNN Zoo fixture directory")
def test_cnn_zoo_loader_and_cli_plan_are_offline(tmp_path, monkeypatch):
    directory = Path(os.environ["WEIGHTSPACE_CNN_ZOO_FIXTURES"]).resolve()
    # Exercise the public extractor's manifest shape even with the local archived copy.
    fixture_dir = tmp_path / "public-fixtures"
    fixture_dir.mkdir()
    archive = json.loads((directory / "manifest.json").read_text())
    for fixture in archive["fixtures"].values():
        for key in ("", "modeldir"):
            fixture["metrics_row"].pop(key, None)
        copyfile(directory / fixture["path"], fixture_dir / fixture["path"])
    copyfile(directory / "layout.csv", fixture_dir / "layout.csv")
    (fixture_dir / "manifest.json").write_text(json.dumps(archive))
    example = Path(__file__).resolve().parents[2] / "examples" / "reading-smoke.json"
    monkeypatch.chdir(tmp_path)
    assert main(["reading", "fixture-manifest", "--fixtures", "public-fixtures", "--output", "panel.json"]) == 0
    assert main(["reading", "plan", "--config", str(example), "--panel", "panel.json", "--output", "plan.json"]) == 0
    plan = json.loads(Path("plan.json").read_text())
    assert plan["planned_calls"] == 30 * 2
    assert plan["panel"]["dataset"]["sampling"] == "fixtures"
    assert len(plan["panel"]["networks"]) == 2
    assert not any("context window" in blocker for blocker in plan["live_preflight_blockers"])
    assert plan["live_preflight_blockers"]
