"""Stage, execute, or summarize the frozen MountainCar comparison."""

import argparse
from dataclasses import asdict
from hashlib import file_digest, sha256
import json
from pathlib import Path
import platform
import shutil
import tarfile
from zipfile import ZipFile

import numpy as np

from .artifacts import RunJournal, canonical_json
from .codex_editor import CATALOG_SOURCE_FILE, catalog_provenance, stage_model_catalog
from .experiment import (
    ExperimentConfig, _prepare_pair, _source_provenance, _transform_policy, planned_cells, run_experiment,
)
from .policies import policy_digest
from .tasks import prepare_task


TASK = "TR02-CTRL-002"
FORMS = ("original", "permutation_relu_scale")
COMPOUND_METHODS = ("compound_random", "compound_greedy", "compound_es")
ARMS = ("no_edit", *COMPOUND_METHODS, "velocity_controller", "codex")
PLAN_PATH = "examples/mountaincar-panel.json"
PROTOCOL_PATH = "docs/mountaincar-panel.md"
FIXTURE_FILES = (
    "manifest.json", "public/policy.npz", "public/failure_trajectory.npz", "private/eval_sets.npz",
)


def _digest(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def _read_json(path):
    return json.loads(path.read_bytes())


def _write(path, contents):
    with path.open("xb") as handle:
        handle.write(contents)


def _write_json(path, record):
    _write(path, canonical_json(record) + b"\n")


def _copy(source, target):
    with source.open("rb") as incoming, target.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing)


def _job_id(fixture):
    return f"fixture-{fixture['fixture_seed']}"


def _configs(plan, root):
    if plan["schema_version"] != 1 or len(plan["fixtures"]) != 5:
        raise ValueError("expected the five-fixture prospective panel")
    jobs = {}
    for fixture in plan["fixtures"]:
        config = ExperimentConfig(
            **{**plan["experiment"], "editor": {
                **plan["experiment"]["editor"], "model_catalog_source": str(root / CATALOG_SOURCE_FILE),
            }},
            fixtures=f"fixtures/{_job_id(fixture)}", fixture_seed=fixture["fixture_seed"], seeds=(fixture["run_seed"],),
        )
        jobs[_job_id(fixture)] = config
    if len(jobs) != 5 or len({config.seeds for config in jobs.values()}) != 5:
        raise ValueError("fixture and paired run seeds must each be distinct")
    bound = 0
    for config in jobs.values():
        if (
            config.tasks != (TASK,) or config.conditions != FORMS or config.methods != ARMS
            or config.fixture_source != "generated"
            or (config.candidate_cap, config.dev_per_region, config.heldout_per_region) != (2, 8, 100)
        ):
            raise ValueError("unexpected task, methods, or evaluator budget")
        options = config.editor_options()
        if options.retries or options.max_calls != 2:
            raise ValueError("expected two native calls per editor cell and zero harness retries")
        bound += options.max_calls * sum(cell["method"] == "codex" for cell in planned_cells(config))
    if bound != plan["max_native_launches"] or bound != 20:
        raise ValueError("invalid panel-wide native launch bound")
    return jobs


def _archive_provenance(source, wheel):
    """Bind the two frozen distributions to the package handling this invocation."""
    with tarfile.open(source, mode="r:") as archive:
        commit = archive.pax_headers.get("comment", "")
        if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
            raise ValueError("expected an unprefixed Git archive of the frozen commit")

        def read(name):
            with archive.extractfile(name) as handle:
                return handle.read()

        archived_modules = {
            name.removeprefix("src/weightspace/"): sha256(read(name)).hexdigest()
            for name in archive.getnames() if name.startswith("src/weightspace/") and name.endswith(".py")
        }
        plan, protocol = read(PLAN_PATH), read(PROTOCOL_PATH)
    with ZipFile(wheel) as archive:
        wheel_modules = {
            name.removeprefix("weightspace/"): sha256(archive.read(name)).hexdigest()
            for name in archive.namelist() if name.startswith("weightspace/") and name.endswith(".py")
        }
    source_provenance = _source_provenance()
    if archived_modules != wheel_modules or wheel_modules != source_provenance["modules_sha256"]:
        raise ValueError("frozen archive, wheel, and loaded package differ")
    return {"git_commit": commit, "package": source_provenance}, plan, protocol


def _runtime():
    import torch

    executable = shutil.which("codex")
    if executable is None:
        raise FileNotFoundError("frozen Codex executable is unavailable on PATH")
    cli = Path(executable).resolve(strict=True)
    return {
        "python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__),
        "platform": platform.platform(), "codex_executable": str(cli), "codex_sha256": _digest(cli),
    }


def _describe_pair(root, config):
    fixture = root / config.fixtures
    task = _prepare_pair(config, fixture, TASK, config.seeds[0])
    transformed, transform = _transform_policy(task.policy, task.spec.sizes, FORMS[1], config.seeds[0])
    return {
        "provenance": task.provenance,
        "starting_policy_sha256": {FORMS[0]: policy_digest(task.policy), FORMS[1]: policy_digest(transformed)},
        "transform": transform,
    }


def prepare_panel(root, *, source, wheel, descriptor, git_commit):
    """Archive inputs and generate each predeclared fixture once; make no editor requests."""
    root = root.resolve()
    provenance, plan_bytes, protocol = _archive_provenance(source, wheel)
    if provenance["git_commit"] != git_commit:
        raise ValueError("source archive differs from the requested frozen Git commit")
    plan = json.loads(plan_bytes)
    jobs = _configs(plan, root)
    runtime = _runtime()
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    _copy(source, root / "source.tar")
    _copy(wheel, root / "package.whl")
    _write(root / "plan.json", plan_bytes)
    _write(root / "protocol.md", protocol)
    archived = _archive_provenance(root / "source.tar", root / "package.whl")
    if archived != (provenance, plan_bytes, protocol):
        raise ValueError("source or wheel changed during staging")
    options = next(iter(jobs.values())).editor_options()
    input_options = type(options)(**{**asdict(options), "model_catalog_source": str(descriptor.resolve())})
    catalog = stage_model_catalog(input_options, root)
    if catalog["catalog_sha256"] != plan["model_catalog_sha256"]:
        raise ValueError("derived catalog differs from the prospective pin")

    preparation = RunJournal.create(root, {"git_commit": git_commit, "jobs": list(jobs)}, run_id="preparation")
    (root / "configs").mkdir()
    pairs = {}
    try:
        for job, config in jobs.items():
            preparation.append("fixture_started", job=job, fixture_seed=config.fixture_seed)
            _write_json(root / "configs" / f"{job}.json", asdict(config))
            prepare_task(root / config.fixtures, TASK, config.fixture_seed)
            pairs[job] = _describe_pair(root, config)
            preparation.append("fixture_prepared", job=job, pair=pairs[job])
    except BaseException as error:
        preparation.append("preparation_failed", job=job, error_type=type(error).__name__, reason=str(error))
        raise
    preparation.append("preparation_completed", fixtures=len(pairs))
    files = ["source.tar", "package.whl", "plan.json", "protocol.md", catalog["source_file"], catalog["catalog_file"]]
    files += [f"configs/{job}.json" for job in jobs]
    files += [f"{config.fixtures}/{TASK}/{name}" for config in jobs.values() for name in FIXTURE_FILES]
    manifest = {
        "schema_version": 1, "source": provenance, "runtime": runtime, "fixtures": pairs,
        "files_sha256": {name: _digest(root / name) for name in files},
    }
    _write_json(root / "frozen-manifest.json", manifest)
    return {"git_commit": git_commit, "prepared_fixtures": len(pairs), "max_native_launches": plan["max_native_launches"]}


def _frozen_plan(root):
    frozen = _read_json(root / "frozen-manifest.json")
    plan_path = root / "plan.json"
    if _digest(plan_path) != frozen["files_sha256"]["plan.json"]:
        raise ValueError("plan differs from the frozen input manifest")
    return frozen, _read_json(plan_path)


def _verify_for_launch(root, frozen, plan, jobs):
    options = next(iter(jobs.values())).editor_options()
    catalog = catalog_provenance(options, root)
    if catalog["catalog_sha256"] != plan["model_catalog_sha256"]:
        raise ValueError("archived catalog differs from the prospective pin")
    for name, expected in frozen["files_sha256"].items():
        if _digest(root / name) != expected:
            raise ValueError(f"frozen artifact changed: {name}")
    provenance, plan_bytes, protocol = _archive_provenance(root / "source.tar", root / "package.whl")
    if (
        frozen["schema_version"] != 1 or frozen["source"] != provenance
        or (root / "plan.json").read_bytes() != plan_bytes or (root / "protocol.md").read_bytes() != protocol
        or _runtime() != frozen["runtime"]
    ):
        raise ValueError("launch inputs or runtime differ from the frozen preparation")
    for job, config in jobs.items():
        if (
            (root / "configs" / f"{job}.json").read_bytes() != canonical_json(asdict(config)) + b"\n"
            or _describe_pair(root, config) != frozen["fixtures"][job]
        ):
            raise ValueError(f"fixture {job} differs from the prospective configuration or frozen pair")


def run_panel(root, *, allow_live=False):
    """Claim this panel once and run the frozen fixture jobs sequentially."""
    if not allow_live:
        raise ValueError("pass --allow-live-editor to execute the native comparison")
    root = root.resolve()
    if (root / "execution").exists():
        raise FileExistsError("this panel already has a launch claim")
    frozen, plan = _frozen_plan(root)
    jobs = _configs(plan, root)
    _verify_for_launch(root, frozen, plan, jobs)
    execution = RunJournal.create(root, {
        "frozen_manifest_sha256": _digest(root / "frozen-manifest.json"),
        "jobs": list(jobs), "max_native_launches": plan["max_native_launches"],
    }, run_id="execution")
    completed = failed = 0
    expected = sum(len(planned_cells(config)) for config in jobs.values())
    stop = None
    try:
        for job, config in jobs.items():
            execution.append("fixture_started", job=job)
            result = run_experiment(config, root / config.fixtures, root / "experiments" / job, allow_live=True)
            execution.append("fixture_terminal", job=job, result=result)
            completed += result["completed"]
            failed += result["failed"]
            if "not_started" in result:
                stop = {"kind": "hard_native_failure", "job": job}
                break
    except BaseException as error:
        execution.append("panel_interrupted", job=job, error_type=type(error).__name__, reason=str(error))
        raise
    terminal = {
        "expected": expected, "completed": completed, "failed": failed,
        "not_started": expected - completed - failed, "stop": stop,
    }
    execution.append("panel_terminal", **terminal)
    return terminal


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="stage the frozen archive and generate the fixture inputs")
    for flag in ("output", "source", "wheel", "descriptor"):
        prepare.add_argument(f"--{flag}", required=True, type=Path)
    prepare.add_argument("--git-commit", required=True)
    run = commands.add_parser("run", help="execute the staged panel once")
    run.add_argument("--input", required=True, type=Path)
    run.add_argument("--allow-live-editor", action="store_true")
    summary = commands.add_parser("summarize", help="audit all planned cells and compute paired contrasts")
    summary.add_argument("--input", required=True, type=Path)
    summary.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_panel(
                args.output, source=args.source, wheel=args.wheel, descriptor=args.descriptor, git_commit=args.git_commit,
            )
        elif args.command == "run":
            result = run_panel(args.input, allow_live=args.allow_live_editor)
        else:
            from .panel_analysis import summarize_panel

            result = summarize_panel(args.input)
            if args.output is not None:
                _write_json(args.output, result)
                return 0
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        parser.exit(2, f"weightspace panel: {error}\n")
    return int(args.command == "run" and result["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
