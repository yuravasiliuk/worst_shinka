from __future__ import annotations

import argparse
import logging
from pathlib import Path
from .config import DEFAULT_INITIAL_MODEL, RunConfig
from .branding import print_logo

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
    run.add_argument("--models", required=True, nargs="+", metavar="MODEL", help="pool of models for bandit selector")
    run.add_argument("--results-dir", type=Path, default=Path("results"), help="root directory for run results")
    run.add_argument("--initial-model", type=Path, default=DEFAULT_INITIAL_MODEL, help="model-0 path")
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

    visualize = subparsers.add_parser("visualize", help="show an evolution tree for a selected run")
    visualize.add_argument("--run-path", required=True, type=Path, help="path to a run directory")
    visualize.add_argument("--save-to", type=Path, help="save a presitent HTML copy at this path, omit for a temporary file")
    visualize.add_argument("--no-open", action="store_true", help="generate HTML without opening a browser")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        if args.command == "run":
            from .orchestrator import run_evolution
            print_logo()
            config = RunConfig(
                models = tuple(args.models),
                results_dir = args.results_dir,
                initial_model = args.initial_model,
                generations = args.generations,
                workers = args.generations,
                parents = args.parents
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
            from .integrations import play_candidate

            model_path = args.model_path.expanduser().resolve()
            if not model_path.exists():
                raise FileNotFoundError(f"Model path does not exists: {model_path}")
            play_candidate(model_path = str(model_path))
            print(f"Play placeholder: model={model_path}")
            return 0

        from . import settings

        if args.config_command == "login":
            path = settings.login()
            print(f"OpenRouter API key has been saved in {path}")
            return 0
        if args.config_command == "logout":
            removed = settings.logout()
            print("OpenRouter API key has been removed." if removed else "No API key was found.")
            if settings.api_key_source():
                print("API key is still provided by the OPENROUTER_API_KEY environment variable.")
            return 0
        print(settings.display_status())
        return 0
    except(FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
