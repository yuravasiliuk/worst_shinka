from __future__ import annotations

from typing import Callable

from worst_shinka.judge import Judge


class BrainstormingJudgeAdapter:

    def __init__(
        self,
        judge: Judge,
        train_function: Callable,
    ):
        self.judge = judge
        self.train_function = train_function

    def evaluate(
        self,
        proposal_1: str,
        proposal_2: str,
        *,
        config_path: str,
        gen_id: int,
        games: int = 5,
    ) -> dict:
        
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            algorithm_1 = tmp_path / "algorithm_1.py"
            algorithm_2 = tmp_path / "algorithm_2.py"

            model_1 = tmp_path / "model_1.pt"
            model_2 = tmp_path / "model_2.pt"

            algorithm_1.write_text(
                proposal_1,
                encoding="utf-8",
            )

            algorithm_2.write_text(
                proposal_2,
                encoding="utf-8",
            )

            self.train_function(
                gen_id=gen_id,
                config_path=config_path,
                algorithm_path=str(algorithm_1),
                model_output_path=str(model_1),
            )

            self.train_function(
                gen_id=gen_id,
                config_path=config_path,
                algorithm_path=str(algorithm_2),
                model_output_path=str(model_2),
            )

            return self.judge.evaluate(
                solution_a=str(model_1),
                solution_b=str(model_2),
                games=games,
                id_a="proposal_1",
                id_b="proposal_2",
            )