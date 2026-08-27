import logging
import os
import sys
from datetime import datetime

import torch
import yaml

GLOBAL_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "global_config.yaml")

with open(GLOBAL_CONFIG_PATH) as _f:
    _global_config = yaml.safe_load(_f)

# Keeps RL_training's INFO-level logs visible when a module here is run
# standalone (e.g. loop_test.py, `python train.py`) instead of through
# worst-shinka's CLI, which already configures the root logger itself
# before RL_training is ever imported.
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

MAX_CYCLES = _global_config["training"]["max_steps_per_episode"]
RESULTS_DIR = os.environ.get("WORST_SHINKA_RESULTS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
TOURNAMENT_TABLE_PATH = os.path.join(RESULTS_DIR, "tournament_table.csv")
MODEL_SCORE_HISTORY_PATH = os.path.join(RESULTS_DIR, "model_score_history.csv")
MODEL_SCORE_HISTORY_HEADER = "generation;elo;score;time_seconds"
RAM_GAMES = {"first_0": 71, "second_0": 72}
ELO_BASELINE = 1200

# Same palette as worst_shinka/cli/terminal.py's ColoredLogFormatter, kept as
# a local copy (not imported) so RL_training stays independent of worst_shinka.
RESET = "\033[0m"
PEACH = "\033[38;2;255;190;140m"
MINT = "\033[38;2;152;255;204m"
BLUE = "\033[94m"
GREEN = "\033[92m"
PURPLE = "\033[95m"


def _model_path(gen_id):
    path = os.path.join(RESULTS_DIR, f"gen_{gen_id}", "model.pt")
    if not os.path.isfile(path):
        raise Exception(f"Path to model {gen_id} doesn't exist")
    return path

def _load_model(path):
    model = torch.load(path, weights_only=False, map_location="cpu")
    model.device = torch.device("cpu")
    model.eval()
    return model


def _load_model_score_history():
    # Files created before the header column existed (or pre-touched empty by
    # worst_shinka's configure_run) have no header line yet - (re)write one via
    # _save_model_score_history so every row after this point is 4 columns.
    if not os.path.isfile(MODEL_SCORE_HISTORY_PATH) or os.path.getsize(MODEL_SCORE_HISTORY_PATH) == 0:
        _save_model_score_history([])
        return []
    with open(MODEL_SCORE_HISTORY_PATH) as f:
        lines = [line.strip() for line in f if line.strip()]
    rows = [line.split(";") for line in lines[1:]]  # lines[0] is the header
    return [
        [
            int(r[0]),
            None if r[1] == "None" else float(r[1]),
            None if r[2] == "None" else float(r[2]),
            None if r[3] == "None" else float(r[3]),
        ]
        for r in rows
    ]


def _save_model_score_history(rows):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(MODEL_SCORE_HISTORY_PATH, "w") as f:
        f.write(MODEL_SCORE_HISTORY_HEADER + "\n")
        for gen_id, elo, avg, duration in rows:
            f.write(
                f"{gen_id};{'None' if elo is None else elo};"
                f"{'None' if avg is None else avg};{'None' if duration is None else duration}\n"
            )


def _styled(text: object, color: str, *, enabled: bool) -> str:
    return f"{color}{text}{RESET}" if enabled else str(text)


def progress_line(label: str, *, label_color: str, generation: int, progress: str, result: str, stream=None) -> None:
    """Writes a status line styled like the CLI's own log lines (colored
    date/time/level prefix). Always prints one plain line per call, one
    below the other, regardless of whether the target is a TTY."""
    target = stream if stream is not None else sys.stderr
    interactive = getattr(target, "isatty", lambda: False)()
    enabled = interactive and not os.getenv("NO_COLOR")

    now = datetime.now().astimezone()
    date = _styled(now.strftime("%Y-%m-%d"), PEACH, enabled=enabled)
    time_ = _styled(now.strftime("%H:%M:%S"), MINT, enabled=enabled)
    level = _styled(f"{'INFO':<8}", BLUE, enabled=enabled)
    label_text = _styled(label, label_color, enabled=enabled)

    line = (
        f"{date} {time_} - {level} - {label_text} "
        f"- Generation: {generation}  Progress: {progress}  Result: {result}"
    )
    target.write(line + "\n")
    target.flush()


def progress_done(*, stream=None) -> None:
    """Kept for backwards compatibility with callers of the old in-place
    progress line; there is no trailing state to clean up anymore."""
    del stream


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"
