from pathlib import Path

from worst_shinka.brainstorming_system.evaluation_adapter import (
    BrainstormingEvaluationAdapter,
)


def test_brainstorming_two_proposals_are_trained_and_judged(monkeypatch):
    """
    Integration test:

        Brainstorming
            ↓
        proposal_1 / proposal_2
            ↓
        training
            ↓
        model_1.pt / model_2.pt
            ↓
        Judge
            ↓
        winner
    """

    trained_models = []

    def fake_train_function(
        *,
        gen_id,
        config_path,
        algorithm_path,
        model_output_path,
    ):
        # Проверяем, что Brainstorming действительно передал
        # два Python-файла в training pipeline.
        assert Path(algorithm_path).exists()

        code = Path(algorithm_path).read_text(encoding="utf-8")

        assert "select_action" in code
        assert "get_epsilon" in code
        assert "select_opponent_action" in code
        assert "update_model" in code

        # Имитируем результат training.
        Path(model_output_path).write_text(
            f"fake model generated from {algorithm_path}",
            encoding="utf-8",
        )

        trained_models.append(model_output_path)

    class FakeJudge:
        def evaluate(
            self,
            solution_a,
            solution_b,
            games,
            id_a=None,
            id_b=None,
        ):
            assert Path(solution_a).exists()
            assert Path(solution_b).exists()

            assert games == 5

            return {
                "winner": solution_a,
                "winner_id": id_a,
                "metrics": {
                    "A": {
                        "win_rate": 0.8,
                        "game_difference": 2.0,
                        "difference_score": 0.7,
                        "consistency": 0.8,
                        "score": 0.77,
                    },
                    "B": {
                        "win_rate": 0.2,
                        "game_difference": -2.0,
                        "difference_score": 0.3,
                        "consistency": 0.2,
                        "score": 0.23,
                    },
                },
                "matches": [
                    {
                        "match": 0,
                        "a_games": 5,
                        "b_games": 2,
                        "winner": "A",
                    }
                ],
            }

    adapter = BrainstormingEvaluationAdapter(
        judge=FakeJudge(),
        train_function=fake_train_function,
    )

    proposal_1 = """
def get_epsilon(episode_index, hyperparameters):
    return hyperparameters["epsilon_end"]


def select_action(model, observation, epsilon, num_actions):
    return 0


def select_opponent_action(model, observation, num_actions):
    return 0


def update_model(model, state, action, reward, next_state, done, hyperparameters):
    model.train_step(
        state,
        action,
        reward,
        next_state,
        done,
        gamma=hyperparameters["gamma"],
    )
"""

    proposal_2 = """
def get_epsilon(episode_index, hyperparameters):
    return hyperparameters["epsilon_end"]


def select_action(model, observation, epsilon, num_actions):
    return num_actions - 1


def select_opponent_action(model, observation, num_actions):
    return num_actions - 1


def update_model(model, state, action, reward, next_state, done, hyperparameters):
    model.train_step(
        state,
        action,
        reward,
        next_state,
        done,
        gamma=hyperparameters["gamma"],
    )
"""

    result = adapter.evaluate(
        proposal_1,
        proposal_2,
        gen_id=1,
        config_path="initial_model/config.yaml",
        games=5,
    )

    # Оба Brainstorming решения должны пройти через training.
    assert len(trained_models) == 2

    # Judge должен вернуть победителя.
    assert result["winner"] is not None

    assert "metrics" in result
    assert "matches" in result

    assert result["metrics"]["A"]["score"] > result["metrics"]["B"]["score"]