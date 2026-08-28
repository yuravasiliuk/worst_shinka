from __future__ import annotations

from typing import Callable

from worst_shinka.judge import Judge
from worst_shinka.brainstorming_system.judge_adapter import (
    BrainstormingJudgeAdapter,
)


class BrainstormingEvaluationAdapter:
    """
    Adapter between Brainstorming and the RL Judge.

    Input:
        Two (algorithm.py, config.yaml) proposal pairs.

    Process:
        proposal + its own config -> training -> model -> Judge.

    Output:
        Judge result containing the selected proposal.
    """

    def __init__(
        self,
        judge: Judge,
        train_function: Callable,
        workers: int = 1,
    ):
        self._adapter = BrainstormingJudgeAdapter(
            judge=judge,
            train_function=train_function,
            workers=workers,
        )

    def evaluate(
        self,
        proposal_1: str,
        proposal_2: str,
        *,
        gen_id: int,
        config_1: dict,
        config_2: dict,
        games: int = 5,
    ) -> dict:
        return self._adapter.evaluate(
            proposal_1=proposal_1,
            proposal_2=proposal_2,
            gen_id=gen_id,
            config_1=config_1,
            config_2=config_2,
            games=games,
        )