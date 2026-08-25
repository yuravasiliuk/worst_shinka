from __future__ import annotations

import argparse
import logging
from pathlib import Path
from .config import DEFAULT_INITIAL_MODEL, MODEL_MODES, RunConfig
from .branding import print_logo
from .terminal import configure_logging, print_login_intro, print_login_success, print_logout_result

log = logging.getLogger(__name__)

def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="worst-shinka", 
        description="Command interface for the Worst-Shinka system."
        )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run the evolution loop")
    run.add_argument("--models", nargs="+", metavar="MODEL", help="pool of models for bandit selector. Deafults to system-assigned models")
    run.add_argument("--mode", choices=MODEL_MODES, default="medium", help="model size and context profile")
    run.add_argument("--results-dir", type=Path, default=Path("results"), help="root directory for run results")
    run.add_argument("--name", help="custom run directory name")
    run.add_argument("--initial-model", type=Path, default=DEFAULT_INITIAL_MODEL, help="directory containing standarized gen_0 inputs")
    run.add_argument("--generations", type=positive_int, default=5, help="number of evolution generations")
    run.add_argument("--workers", type=positive_int, default=1, help="maximum concurrent workers")
    run.add_argument("--parents", type=positive_int, default=4, help="number of parents requested per evolution")
   
    config = subparsers.add_parser("config", help="show setings and manage the OpenRouter API key")
    config_sub = config.add_subparsers(dest="config_command")
    config_sub.add_parser("login", help="save or replace the OpenRouter API key")
    config_sub.add_parser("status", help="show configuration and credential status")
    config_sub.add_parser("logout", help="remove OpenRouter credentials")

    play = subparsers.add_parser("play", help="launch a game from a selected result run")
    play.add_argument("--model-path", required=True, type=Path, help="path to the model against which the user wants to play")
    play.add_argument("--opponent-path", type=Path, help="second model for bot-vs-bot match")


    reset = subparsers.add_parser("reset", help = "reset training progress for a selected run")
    reset.add_argument("--run-path", required=True, type=Path, help = "reset training progress for a selected run")

    visualize = subparsers.add_parser("visualize", help="show an evolution tree for a selected run")
    visualize.add_argument("--run-path", required=True, type=Path, help="path to a run directory")
    visualize.add_argument("--save-to", type=Path, help="save a presitent HTML copy at this path, omit for a temporary file")
    visualize.add_argument("--no-open", action="store_true", help="generate HTML without opening a browser")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    #logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    configure_logging()
    try:
        if args.command == "run":
            from .orchestrator import run_evolution
            print_logo()
            config = RunConfig(
                models = tuple(args.models) if args.models is not None else None,
                results_dir = args.results_dir,
                name = args.name,
                initial_model = args.initial_model,
                generations = args.generations,
                workers = args.workers,
                parents = args.parents,
                mode = args.mode
            )
            run_dir = run_evolution(config)
            print(f"Run directory: {run_dir}")
            print("Placeholders; integrations.py")
            return 0

        if args.command == "visualize":
            from .visualization import visualize_run

            output = visualize_run(
                run_path = args.run_path,
                save_to = args.save_to,
                open_browser = not args.no_open
            )
            print(f"Visualization: {output}")
            return 0

        if args.command == "play":
            from .play import run_game

            model_path = args.model_path.expanduser().resolve()
            if not model_path.exists():
                raise FileNotFoundError(f"Model path does not exists: {model_path}")

            opponent_path = None
            if args.opponent_path is not None:
                opponent_path = args.opponent_path.expanduser().resolve()
                if not opponent_path.exists():
                    raise FileNotFoundError(f"Opponent model path does not exists: {opponent_path}")

            return run_game(model_path, opponent_path)

        if args.command == "reset":
            from .integrations import reset_run

            run_path = args.run_path.expanduser().resolve()
            if not run_path.is_dir():
                raise FileNotFoundError(f"Run path does not exist: {run_path}")

            reset_run(run_dir=run_path)
            log.warning(f"Reset run directory: {run_path}")

        from . import settings

        if args.config_command == "login":
            print_login_intro(settings.credentials_path())
            path = settings.login()
            print_login_success(path)
            return 0
        if args.config_command == "logout":
            removed = settings.logout()
            print_logout_result(removed=removed, still_connected=settings.api_key_source() is not None)
            return 0
        print(settings.display_status())
        return 0
    except(FileExistsError, FileNotFoundError, ValueError, RuntimeError) as exc:
        log.error("%s", exc)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())