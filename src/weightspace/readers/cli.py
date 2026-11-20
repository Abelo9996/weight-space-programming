"""CLI registration for CNN reading, with lazy optional imports."""

from pathlib import Path


def add_commands(commands):
    reading = commands.add_parser("reading", help="plan and analyze CNN accuracy reading experiments")
    actions = reading.add_subparsers(dest="reading_action", required=True)
    fixture = actions.add_parser("fixture-manifest", help="make an offline manifest for the two CNN Zoo fixtures")
    fixture.add_argument("--fixtures", type=Path, required=True)
    fixture.add_argument("--output", type=Path, required=True)
    plan = actions.add_parser("plan", help="verify inputs and estimate requests offline")
    run = actions.add_parser("run", help="execute a funded first-party API comparison")
    for command in (plan, run):
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--panel", type=Path, required=True)
        command.add_argument("--output", type=Path, required=command is run)
    run.add_argument("--allow-api", action="store_true")
    run.add_argument("--confirm-credits", action="store_true")
    run.add_argument("--approve-over-40", action="store_true", help="record approval for a projection above USD 40")
    summary = actions.add_parser("summarize", help="score a reader journal offline")
    summary.add_argument("--input", type=Path, required=True)
    summary.add_argument("--output", type=Path)
    summary.add_argument("--bootstrap-replicates", type=int, default=2000)
    summary.add_argument("--bootstrap-seed", type=int, default=20261119)
    baseline = actions.add_parser("baseline", help="fit train-only weight statistics and score the panel offline")
    baseline.add_argument("--panel", type=Path, required=True)
    baseline.add_argument("--seed", type=int, default=20261119)
    baseline.add_argument("--output", type=Path)


def dispatch(args):
    result, output, status = _dispatch(args)
    from .experiment import reject_embedded_credentials
    reject_embedded_credentials(result)
    return result, output, status


def _dispatch(args):
    action = args.reading_action
    if action == "fixture-manifest":
        from .data import fixture_manifest
        return fixture_manifest(args.fixtures, args.output), args.output, 0
    if action == "summarize":
        from .analysis import summarize
        return summarize(args.input, bootstrap_replicates=args.bootstrap_replicates,
                         bootstrap_seed=args.bootstrap_seed), args.output, 0
    if action == "baseline":
        from .baseline import run_baseline
        return run_baseline(args.panel, seed=args.seed), args.output, 0
    if action == "run" and not (args.allow_api and args.confirm_credits):
        raise ValueError("reading run requires --allow-api and --confirm-credits")
    from .experiment import load_config, prepare_plan, run_plan
    plan = prepare_plan(load_config(args.config), args.panel)
    if action == "plan":
        return plan.metadata, args.output, 0
    result = run_plan(plan, args.output, allow_api=args.allow_api, confirm_credits=args.confirm_credits,
                      approve_over_40=args.approve_over_40)
    return result, None, int(result["failed"] > 0 or result["unattempted"] > 0 or result["stop_reason"] is not None)
