from __future__ import annotations
from typing import Any

def fetch_candidates(*, limit: int) -> list[dict[str, Any]]:
    #TODO(Dawid) - trained candidates, scores, logs, turnament table
    # id, parent_id, score
    return []

def train_and_evaluate(*, proposals: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    return []

def select_models_with_bandit(*, models: tuple[str, ...], count: int = 2) -> list[str]:
    return list(models[:count])

def evolve_with_models(*, models: list[str], parents: list[dict[str, Any]], generation: int) -> list[dict[str, Any]]:
    return []

def judge_candidates(*, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return []

def play_candidate(*, model_path: str) -> None:
    return None


"""pliki od dawida:
- run_play.py w tym funkcja run_play ktora odpala wizualizacje i gre,
- run_turnament.py 
- run_training.py
- run_agreggate_data.py
- run_reset.py
"""