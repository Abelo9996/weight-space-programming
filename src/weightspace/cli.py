"""Command line for preparing, running and summarizing policy experiments."""

import argparse
import json
from pathlib import Path
import sys

from .analysis import summarize
from .experiment import load_config, run_experiment
from .tasks import TASKS, prepare_task


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="generate local policy fixtures")
    prepare.add_argument("--task", action="append", required=True, choices=tuple(TASKS))
    prepare.add_argument("--seed", required=True, type=int)
    prepare.add_argument("--output", required=True, type=Path)

    run = commands.add_parser("run", help="run an experiment into a new directory")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--fixtures", type=Path, help="override the fixture directory in the config")
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--allow-live-editor", action="store_true")

    summary = commands.add_parser("summarize", help="summarize paired policy experiment results")
    summary.add_argument("--input", required=True, type=Path)
    summary.add_argument("--output", type=Path)
    summary.add_argument("--bootstrap-replicates", type=int, default=2000)
    summary.add_argument("--bootstrap-seed", type=int, default=20261119)
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    output_file = None
    try:
        if args.command == "prepare":
            result = {task: str(prepare_task(args.output, task, args.seed)) for task in dict.fromkeys(args.task)}
        elif args.command == "run":
            config = load_config(args.config)
            fixtures = args.fixtures if args.fixtures is not None else args.config.parent / config.fixtures
            result = run_experiment(config, fixtures, args.output, allow_live=args.allow_live_editor)
        else:
            result = summarize(
                args.input, bootstrap_replicates=args.bootstrap_replicates, bootstrap_seed=args.bootstrap_seed,
            )
            output_file = args.output
        rendered = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
        if output_file is None:
            sys.stdout.write(rendered)
        else:
            with output_file.open("x", encoding="utf-8") as handle:
                handle.write(rendered)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        parser.exit(2, f"weightspace: {error}\n")
    return int(args.command == "run" and result["failed"] > 0)
