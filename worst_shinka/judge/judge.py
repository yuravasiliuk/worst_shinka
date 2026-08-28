from __future__ import annotations

import json
import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import importlib.util
from pathlib import Path

def _load_rl_training_utils():
    utils_path = Path(__file__).resolve().parents[2] / "RL_training" / "utils.py"
    spec = importlib.util.spec_from_file_location("worst_shinka._rl_training_utils", utils_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_rl_utils = _load_rl_training_utils()
MAX_CYCLES = _rl_utils.MAX_CYCLES
RAM_GAMES = _rl_utils.RAM_GAMES
_load_model = _rl_utils._load_model


def _play_match_task(solution_a: str, solution_b: str, match_index: int) -> dict[str, Any]:
    model_a = _load_model(solution_a)
    model_b = _load_model(solution_b)
    judge = Judge()
    if match_index % 2 == 0:
        games_a, games_b = judge._play_match(model_a, model_b)
    else:
        games_b, games_a = judge._play_match(model_b, model_a)

    if games_a > games_b:
        winner = "A"
    elif games_b > games_a:
        winner = "B"
    else:
        winner = "draw"

    return {
        "match": match_index,
        "a_games": games_a,
        "b_games": games_b,
        "winner": winner,
    }

@dataclass
class JudgeConfig:
    win_rate_weight: float = 0.60
    game_difference_weight: float = 0.25
    consistency_weight: float = 0.15

    def __post_init__(self) -> None:
        weights = (
            self.win_rate_weight,
            self.game_difference_weight,
            self.consistency_weight,
        )

        if any(w < 0 for w in weights):
            raise ValueError("Judge weights must be non-negative")

        if abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError("Judge weights must sum to 1")


class Judge:


    def __init__(
        self,
        config: Optional[JudgeConfig] = None,
        train_function=None,
        history_path: Optional[str] = None,
        workers: int = 1,
    ):
        self.config = config or JudgeConfig()
        self.train_function = train_function
        self.workers = max(1, workers)

        self.history_path = Path(
            history_path
            or "worst_shinka/judge/history/winners.jsonl"
        )

        self.history_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def _difference_score(game_difference: float) -> float:
        score = round(0.5 + 0.1 * game_difference, 10)
        return max(0.0, min(1.0, score))

    @staticmethod
    def _consistency_score(
        results: list[dict[str, Any]],
        candidate: str,
    ) -> float:
        if not results:
            return 0.5

        values = []

        for result in results:
            winner = result["winner"]

            if winner == candidate:
                values.append(1.0)
            elif winner == "draw":
                values.append(0.5)
            else:
                values.append(0.0)

        return sum(values) / len(values)

    @staticmethod
    def _select_action(observation, model) -> int:
        import torch

        with torch.no_grad():
            obs = torch.as_tensor(
                observation,
                dtype=torch.float32,
            ).unsqueeze(0)

            q_values = model(obs)

            if q_values.ndim > 1:
                q_values = q_values[0]

            return int(torch.argmax(q_values).item())

    def _play_match(self, model_a, model_b) -> tuple[int, int]:
        from pettingzoo.atari import tennis_v3

        env = tennis_v3.env(
            render_mode=None,
            obs_type="ram",
            max_cycles=MAX_CYCLES,
        )

        env.reset()

        models = {
            "first_0": model_a,
            "second_0": model_b,
        }

        for agent in env.agent_iter():
            observation, _reward, termination, truncation, _info = env.last()

            if termination or truncation:
                action = None
            else:
                action = self._select_action(
                    observation,
                    models[agent],
                )

            env.step(action)

        ram = env.unwrapped.ale.getRAM()

        games_a = int(ram[RAM_GAMES["first_0"]])
        games_b = int(ram[RAM_GAMES["second_0"]])

        env.close()

        return games_a, games_b

    def _evaluate_matches(
        self,
        solution_a: str,
        solution_b: str,
        games: int,
    ) -> list[dict[str, Any]]:
        if self.workers == 1:
            return [_play_match_task(solution_a, solution_b, match_index) for match_index in range(games)]

        context = multiprocessing.get_context("spawn")
        tasks = [(solution_a, solution_b, match_index) for match_index in range(games)]
        with context.Pool(processes=min(self.workers, games)) as pool:
            return pool.starmap(_play_match_task, tasks)

    def _record_winner(
        self,
        result: dict[str, Any],
        *,
        generation: Optional[int] = None,
        program_a: Optional[str] = None,
        program_b: Optional[str] = None,
    ) -> None:
        

        winner_id = result.get("winner_id")

        # No winner = draw.
        if winner_id is None:
            return

        if winner_id == result.get("id_a"):
            winner_side = "A"
            winner_code = program_a
            winner_metrics = result["metrics"]["A"]

        elif winner_id == result.get("id_b"):
            winner_side = "B"
            winner_code = program_b
            winner_metrics = result["metrics"]["B"]

        else:
            winner_side = None
            winner_code = None
            winner_metrics = None

        record = {
            "generation": generation,
            "winner_id": winner_id,
            "winner_side": winner_side,
            "winner_model": result.get("winner"),
            "winner_code": winner_code,
            "score": (
                winner_metrics["score"]
                if winner_metrics is not None
                else None
            ),
            "metrics": result.get("metrics", {}),
            "matches": result.get("matches", []),
        }

        with self.history_path.open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    def evaluate(
        self,
        solution_a: str,
        solution_b: str,
        games: int = 5,
        *,
        id_a: Optional[str] = None,
        id_b: Optional[str] = None,
        program_a: Optional[str] = None,
        program_b: Optional[str] = None,
        generation: Optional[int] = None,
    ) -> dict[str, Any]:

        if games <= 0:
            raise ValueError("games must be greater than zero")

        results = self._evaluate_matches(
            solution_a,
            solution_b,
            games,
        )

        wins_a = sum(
            r["winner"] == "A"
            for r in results
        )

        wins_b = sum(
            r["winner"] == "B"
            for r in results
        )

        draws = sum(
            r["winner"] == "draw"
            for r in results
        )

        win_rate_a = (
            wins_a + 0.5 * draws
        ) / games

        win_rate_b = (
            wins_b + 0.5 * draws
        ) / games

        average_difference = sum(
            r["a_games"] - r["b_games"]
            for r in results
        ) / games

        difference_score_a = self._difference_score(
            average_difference
        )

        difference_score_b = round(
            1.0 - difference_score_a,
            10,
        )

        consistency_a = self._consistency_score(
            results,
            "A",
        )

        consistency_b = self._consistency_score(
            results,
            "B",
        )

        score_a = (
            self.config.win_rate_weight * win_rate_a
            + self.config.game_difference_weight
            * difference_score_a
            + self.config.consistency_weight
            * consistency_a
        )

        score_b = (
            self.config.win_rate_weight * win_rate_b
            + self.config.game_difference_weight
            * difference_score_b
            + self.config.consistency_weight
            * consistency_b
        )

        if score_a > score_b:
            winner = solution_a
            winner_id = id_a

        elif score_b > score_a:
            winner = solution_b
            winner_id = id_b

        else:
            winner = None
            winner_id = None

        result = {
            "winner": winner,
            "winner_id": winner_id,
            "id_a": id_a,
            "id_b": id_b,
            "metrics": {
                "A": {
                    "win_rate": win_rate_a,
                    "game_difference": average_difference,
                    "difference_score": difference_score_a,
                    "consistency": consistency_a,
                    "score": score_a,
                },
                "B": {
                    "win_rate": win_rate_b,
                    "game_difference": -average_difference,
                    "difference_score": difference_score_b,
                    "consistency": consistency_b,
                    "score": score_b,
                },
            },
            "matches": results,
        }

        self._record_winner(
            result,
            generation=generation,
            program_a=program_a,
            program_b=program_b,
        )

        return result

    def evaluate_programs(
        self,
        program_a: str,
        program_b: str,
        *,
        gen_id: int,
        config_path: str,
        games: int = 5,
        id_a: Optional[str] = None,
        id_b: Optional[str] = None,
    ) -> dict[str, Any]:

        if self.train_function is None:
            raise RuntimeError(
                "train_function is required for evaluate_programs()"
            )

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            algorithm_a = tmp_path / "algorithm_a.py"
            algorithm_b = tmp_path / "algorithm_b.py"

            model_a = tmp_path / "model_a.pt"
            model_b = tmp_path / "model_b.pt"

            algorithm_a.write_text(
                program_a,
                encoding="utf-8",
            )

            algorithm_b.write_text(
                program_b,
                encoding="utf-8",
            )

            self.train_function(
                gen_id=gen_id,
                config_path=config_path,
                algorithm_path=str(algorithm_a),
                model_output_path=str(model_a),
            )

            self.train_function(
                gen_id=gen_id,
                config_path=config_path,
                algorithm_path=str(algorithm_b),
                model_output_path=str(model_b),
            )

            result = self.evaluate(
                solution_a=str(model_a),
                solution_b=str(model_b),
                games=games,
                id_a=id_a,
                id_b=id_b,
                program_a=program_a,
                program_b=program_b,
                generation=gen_id,
            )

            result["input_type"] = "python_programs"
            result["generation"] = gen_id

            result["programs"] = {
                "A": program_a,
                "B": program_b,
            }

            return result