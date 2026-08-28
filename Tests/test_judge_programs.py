from pathlib import Path

from worst_shinka.judge import Judge


def test_judge_evaluates_two_python_programs(tmp_path):
    trained_models = {}

    def fake_train(
        *,
        gen_id,
        config_path,
        algorithm_path,
        model_output_path,
    ):
        code = Path(algorithm_path).read_text(encoding="utf-8")

        # Проверяем, что Judge действительно передал Python-программу
        assert "def select_action" in code

        Path(model_output_path).write_text(
            "fake model",
            encoding="utf-8",
        )

        trained_models[model_output_path] = code

    class FakeJudge(Judge):
        def _evaluate_matches(
            self,
            solution_a,
            solution_b,
            games,
        ):
            assert Path(solution_a).exists()
            assert Path(solution_b).exists()

            return [
                {
                    "match": 0,
                    "a_games": 5,
                    "b_games": 2,
                    "winner": "A",
                },
                {
                    "match": 1,
                    "a_games": 4,
                    "b_games": 3,
                    "winner": "A",
                },
                {
                    "match": 2,
                    "a_games": 3,
                    "b_games": 3,
                    "winner": "draw",
                },
            ]

    judge = FakeJudge(train_function=fake_train)

    program_a = """
def get_epsilon(episode_index, hyperparameters):
    return 0.1

def select_action(model, observation, epsilon, num_actions):
    return 0

def select_opponent_action(model, observation, num_actions):
    return 0

def update_model(model, state, action, reward, next_state, done, hyperparameters):
    pass
"""

    program_b = """
def get_epsilon(episode_index, hyperparameters):
    return 0.9

def select_action(model, observation, epsilon, num_actions):
    return 1

def select_opponent_action(model, observation, num_actions):
    return 1

def update_model(model, state, action, reward, next_state, done, hyperparameters):
    pass
"""

    result = judge.evaluate_programs(
        program_a,
        program_b,
        gen_id=1,
        config_path="initial_model/config.yaml",
        games=3,
        id_a="proposal-A",
        id_b="proposal-B",
    )

    assert result["input_type"] == "python_programs"
    assert result["generation"] == 1

    assert result["winner_id"] == "proposal-A"

    assert result["metrics"]["A"]["win_rate"] == 5 / 6
    assert result["metrics"]["B"]["win_rate"] == 1 / 6

    assert len(result["matches"]) == 3

    assert len(trained_models) == 2