from __future__ import annotations

import importlib
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Any

RL_RESULTS_ENV = "WORST_SHINKA_RESULTS_DIR"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RL_TRAINING_DIR = PROJECT_ROOT / "RL_training"

_run_dir: Path | None = None


def configure_run(run_dir: Path) -> Path:
    global _run_dir
    _run_dir = run_dir.expanduser().resolve()
    _run_dir.mkdir(parents=True, exist_ok=True)
    os.environ[RL_RESULTS_ENV] = str(_run_dir)

    for name in ("tournament_table.csv", "model_score_history.csv"):
        (_run_dir / name).touch(exist_ok=True)
    return _run_dir


def _require_run() -> Path:
    if _run_dir is None:
        raise RuntimeError("RL integration has not been configured with a run directory")
    return _run_dir

def _rl_modules() -> dict[str, Any]:
    run_dir = _require_run()
    if not RL_TRAINING_DIR.is_dir():
        raise FileNotFoundError(f"RL_training directory does not exist: {RL_TRAINING_DIR}")
    rl_path = str(RL_TRAINING_DIR)
    if rl_path not in sys.path:
        sys.path.insert(0, rl_path)

    utils = importlib.import_module("utils")
    utils.RESULTS_DIR = str(run_dir)
    utils.TURNAMENT_TABLE_PATH = str(run_dir / "tournament_table.csv")
    utils.MODEL_SCORE_HISTORY_PATH = str(run_dir / "model_score_history.csv")

    modules: dict[str, Any] = {"utils": utils}
    for name in ("run_training", "run_tournament", "run_aggregate_data", "run_play", "run_reset"):
        modules[name] = importlib.import_module(name)
    modules["run_training"].RESULTS_DIR = str(run_dir)
    modules["run_tournament"].RESULTS_DIR = str(run_dir)
    modules["run_tournament"].TOURNAMENT_TABLE_PATH = str(run_dir / "tournament_table.csv")
    modules["run_aggregate_data"].RESULTS_DIR = str(run_dir)
    modules["run_reset"].RESULTS_DIR = str(run_dir)
    return modules


def _generation_number(generation_dir: Path) -> int:
    try:
        return int(generation_dir.name.removeprefix("gen_"))
    except ValueError as exc:
        raise ValueError(f"generation directory must be named gen_N: {generation_dir}") from exc

def _standard_generation_files(generation_dir: Path) -> tuple[Path, Path, Path]:
    directory = generation_dir.expanduser().resolve()
    if directory.parent != _require_run() or not directory.name.startswith("gen_"):
        raise ValueError(f"generation dirctory must be directly inside {_require_run()}: {directory}")
    config = directory / "config.yaml"
    algorithm = directory / "algorithm.py"
    model = directory / "model.pt"

    for label, path in (("config", config), ("algorithm", algorithm)):
        if not path.is_file():
            raise FileNotFoundError(f"Missing standarized {label} file: {path}")

    return config, algorithm, model

def prepare_initial_generation(*, source_dir: Path, generation_dir: Path) -> None:
    src = source_dir.expanduser().resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"Initial model directory does not exist: {src}")
    generation_dir.mkdir(parents=True, exist_ok=True)
    for name in ("config.yaml", "algorithm.py", "model.pt", "training_logs.txt"):
        source_file = src / name
        if source_file.is_file():
            shutil.copy2(source_file, generation_dir / name)
    _standard_generation_files(generation_dir)
    if (generation_dir / "model.pt").is_file():
        (generation_dir / "training_logs.txt").touch(exist_ok=True)



def fetch_candidates(*, limit: int) -> list[dict[str, Any]]:
    rows = _rl_modules()["utils"]._load_model_score_history()
    candidates = []
    for gen, elo, score in reversed(rows):
        model = _require_run() / f"gen_{gen}" / "model.pt"
        candidates.append({
            "id": f"model-{gen}",
            "generation": gen,
            "model": str(model),
            "score": score, 
            "elo": elo,
            "staus": "correct" if model.is_file() else "incorrect"
        })
        return candidates[:limit]


def _train_generation(generation_dir: Path) -> dict[str, Any]:
    conf, alg, model = _standard_generation_files(generation_dir)
    gen = _generation_number(generation_dir)
    modules = _rl_modules()

    if not model.is_file():
        with tempfile.TemporaryDirectory(prefix="worst-shinka-") as temp:
            temp_dir = Path(temp)
            conf_input = temp_dir / "config.yaml"
            alg_input = temp_dir / "algorithm.py"
            shutil.copy2(conf, conf_input)
            shutil.copy2(alg, alg_input)
            modules["run_training"].run_training(gen, str(conf_input), str(alg_input))

    modules["run_tournament"].run_tournament(gen)
    agg = modules["run_aggregate_data"].run_aggregate_data(gen)

    return _candidate_from_aggregate(gen, model, agg)


def _candidate_from_aggregate(generation: int, model: Path, aggregate: dict[str, Any]) -> dict[str, Any]:
    elo = score = None
    for line in str(aggregate.get("model_score_history") or "").splitlines():
        fields = line.split(";")
        if len(fields) >= 3 and fields[0] == str(generation):
            elo = None if fields[1] == "None" else float(fields[1])
            score = None if fields[2] == "None" else float(fields[2])
            break

    return{
        "id": f"model-{generation}",
        "parent_id": None if generation == 0 else f"model-{generation-1}",
        "generation": generation,
        "model": str(model),
        "elo": elo,
        "status": "correct"
    }    

def train_and_evaluate(*, proposals: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    del workers
    evaluated = []
    for proposal in proposals:
        dir_val = proposal.get("generation_dir")
        if dir_val is None:
            gen = proposal.get("generation")
            if gen is None:
                raise ValueError("Proposal must contain generation_dir or generation")
            dir_val = _require_run() / f"gen_{gen}"
        candidate = _train_generation(Path(dir_val))
        candidate.update({key: value for key, value in proposal.items() if key not in candidate})
        evaluated.append(candidate)

    return evaluated

def select_models_with_bandit(*, models: tuple[str, ...], count: int = 2) -> list[str]:
    return list(models[:count])

def evolve_with_models(*, models: list[str], parents: list[dict[str, Any]], generation: int) -> list[dict[str, Any]]:
    del models, parents, generation

    return []

def judge_candidates(*, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [candidate for candidate in candidates if candidate.get("status") == "correct"]

def _model_location(model_path: str) -> tuple[Path, int]:
    path = Path(model_path).expanduser().resolve()
    if path.name != "model.pt" or not path.parent.name.startswith("gen_"):
        raise ValueError("model must be a <run>/<gen_N>/model.pt file")
    if not path.is_file():
        raise FileNotFoundError(f"Model does not exist: {path}")
    return path, _generation_number(path.parent)

def play_candidate(*, model_path: str, opponent_path: str | None = None) -> None:
    model, gen = _model_location(model_path)
    configure_run(model.parent.parent)
    if opponent_path is not None:
        opponent, opponent_gen = _model_location(opponent_path)
        if opponent.parent.parent != model.parent.parent:
            raise ValueError("Both bot models must belong to the same run directory") #update in future not to
        _rl_modules()["run_play"].run_play(gen, opponent_gen)
        return
    _play_human_vs_model(model)


def _human_action(pygame: Any) -> int:
    keys = pygame.key.get_pressed()
    up, down, left, right = keys[pygame.K_UP], keys[pygame.K_DOWN], keys[pygame.K_LEFT], keys[pygame.K_RIGHT]
    fire = keys[pygame.K_SPACE]
    direction = ""

    if up and right:
        direction = "UPRIGHT"
    elif up and left:
        direction = "UPLEFT"
    elif down and right:
        direction = "DOWNRIGHT"
    elif down and left:
        direction = "DOWNLEFT"
    elif up:
        direction = "UP"
    elif down:
        direction = "DOWN"
    elif right:
        direction = "RIGHT"
    elif left:
        direction = "LEFT"
    action_names = ["NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT", 
                    "DOWNRIGHT", "DOWNLEFT", "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE", 
                    "UPRIGHTFIRE", "UPLEFTFIRE", "DOWNRIGHTFIRE", "DOWNLEFTFIRE"]
    name = f"{direction}FIRE" if direction and fire else direction or ("FIRE" if fire else "NOOP")

    return action_names.index(name)

def _play_human_vs_model(model_path: Path) -> None:
    import pygame
    import torch
    from pettingzoo.atari import tennis_v3

    modules = _rl_modules()
    model = modules["utils"]._load_model(str(model_path))
    env = tennis_v3.env(render_mode="rgb_array", obs_type = "ram", max_cycles=modules["utils"].MAX_CYCLES)
    env.reset()
    ale = env.unwrapped.ale
    crop_top, crop_left, zoom, fps = 4, 8, 4, 30
    screen_h = ale.getScreenDims()[1] - crop_top
    screen_w = ale.getScreenDims()[0] - crop_left
    pygame.display.init()
    screen = pygame.display.set_mode((screen_w * zoom, screen_h * zoom))
    clock = pygame.time.Clock()

    running = True

    try:
        for agent in env.agent_iter():
            for event in pygame.event.get():
                if event.type == pygame.QUIT or(event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    running = False
            if not running:
                break

            observation, _reward, termination, truncation, _info = env.last()
            done = termination or truncation
            is_last = bool(env.agents) and agent == env.agents[-1]
            if done: 
                action = None
            elif agent == "first_0":
                action = _human_action(pygame)
            elif agent == "second_0":
                with torch.no_grad():
                    tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
                    action = int(torch.argmax(model(tensor), dim=-1).item())
            env.step(action)
            if is_last:
                frame = ale.getScreenRGB()[crop_top:, crop_left:]
                surface = pygame.image.frombuffer(frame.tobytes(), frame.shape[:2][::-1], "RGB")
                surface = pygame.transform.scale(surface, (screen_w * zoom, screen_h * zoom))
                screen.blit(surface, (0, 0))
                pygame.display.flip()
                clock.tick(fps)
    finally:
        env.close()
        pygame.quit()

def reset_run(*, run_dir: Path) -> None:
    resolved = configure_run(run_dir)
    _rl_modules()["run_reset"].run_reset()
    for path in (
        resolved / "run.json",
        resolved / "lineage.json",
        resolved / "gen_0" / "metrics.json",
        resolved / "gen_0" / "solutions.json",
        resolved / "gen_0" / "lineage.json"
    ):
        if path.is_file():
            path.unlink()