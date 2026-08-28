from __future__ import annotations

import multiprocessing
import os
from typing import Callable

from worst_shinka.judge import Judge


class BrainstormingJudgeAdapter:

    def __init__(
        self,
        judge: Judge,
        train_function: Callable,
        workers: int = 1,
    ):
        self.judge = judge
        self.train_function = train_function
        self.workers = max(1, workers)

    def _train_or_none(self, gen_id: int, config_path: str, algorithm_path, model_output_path) -> "str | None":
        """Train one proposal; return None on success or an error summary on failure.

        Static pre-flight checks (missing functions, unknown config keys, forbidden
        assert/try-except) catch most broken proposals before wasting a training run, but
        they can't enumerate every way an LLM might misuse the model API (e.g. calling a
        list method on the Q-value ndarray). This is the failsafe: any proposal whose
        training crashes for any reason is treated as broken and omitted, rather than
        taking down the whole run.
        """
        try:
            os.environ["WORST_SHINKA_SKIP_HISTORY"] = "1"
            self.train_function(
                gen_id=gen_id,
                config_path=config_path,
                algorithm_path=str(algorithm_path),
                model_output_path=str(model_output_path),
            )
            return None
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"
        finally:
            os.environ.pop("WORST_SHINKA_SKIP_HISTORY", None)

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

            training_args = [
                (gen_id, config_path, algorithm_1, model_1),
                (gen_id, config_path, algorithm_2, model_2),
            ]
            if self.workers == 1:
                errors = [self._train_or_none(*args) for args in training_args]
            else:
                context = multiprocessing.get_context("spawn")
                with context.Pool(processes=min(self.workers, len(training_args))) as pool:
                    errors = pool.starmap(self._train_or_none, training_args)
            error_1, error_2 = errors

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