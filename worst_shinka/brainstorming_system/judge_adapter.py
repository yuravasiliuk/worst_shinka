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

    def _train_or_none(self, *, gen_id: int, config_path, algorithm_path, model_output_path) -> "str | None":
        """Train one proposal; return None on success or an error summary on failure.

        Static pre-flight checks (missing functions, unknown config keys, forbidden
        assert/try-except) catch most broken proposals before wasting a training run, but
        they can't enumerate every way an LLM might misuse the model API (e.g. calling a
        list method on the Q-value ndarray). This is the failsafe: any proposal whose
        training crashes for any reason is treated as broken and omitted, rather than
        taking down the whole run.
        """
        try:
            self.train_function(
                gen_id=gen_id,
                config_path=str(config_path),
                algorithm_path=str(algorithm_path),
                model_output_path=str(model_output_path),
            )
            return None
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"

    def evaluate(
        self,
        proposal_1: str,
        proposal_2: str,
        *,
        config_1: dict,
        config_2: dict,
        gen_id: int,
        games: int = 5,
    ) -> dict:

        import tempfile
        import yaml
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            algorithm_1 = tmp_path / "algorithm_1.py"
            algorithm_2 = tmp_path / "algorithm_2.py"

            config_path_1 = tmp_path / "config_1.yaml"
            config_path_2 = tmp_path / "config_2.yaml"

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

            config_path_1.write_text(yaml.safe_dump(config_1, sort_keys=False), encoding="utf-8")
            config_path_2.write_text(yaml.safe_dump(config_2, sort_keys=False), encoding="utf-8")

            error_1 = self._train_or_none(
                gen_id=gen_id, config_path=config_path_1, algorithm_path=algorithm_1, model_output_path=model_1
            )
            error_2 = self._train_or_none(
                gen_id=gen_id, config_path=config_path_2, algorithm_path=algorithm_2, model_output_path=model_2
            )

            if error_1 and error_2:
                return {
                    "winner": None,
                    "winner_id": None,
                    "id_a": "proposal_1",
                    "id_b": "proposal_2",
                    "metrics": {},
                    "matches": [],
                    "training_error": (
                        f"Both proposals failed to train and were omitted. "
                        f"Proposal 1: {error_1} Proposal 2: {error_2}"
                    ),
                }

            if error_1 or error_2:
                failed_id, winner_id = ("proposal_1", "proposal_2") if error_1 else ("proposal_2", "proposal_1")
                winner_model = model_2 if error_1 else model_1
                return {
                    "winner": str(winner_model),
                    "winner_id": winner_id,
                    "id_a": "proposal_1",
                    "id_b": "proposal_2",
                    "metrics": {},
                    "matches": [],
                    "training_error": f"{failed_id} failed to train and was omitted: {error_1 or error_2}",
                }

            return self.judge.evaluate(
                solution_a=str(model_1),
                solution_b=str(model_2),
                games=games,
                id_a="proposal_1",
                id_b="proposal_2",
            )