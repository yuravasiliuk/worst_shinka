from worst_shinka.brainstorming_system.evaluation_adapter import (
    BrainstormingEvaluationAdapter,
)


def test_brainstorming_evaluation_adapter_trains_two_proposals_and_judges():
    trained = []

    def fake_train_function(
        *,
        gen_id,
        config_path,
        algorithm_path,
        model_output_path,
    ):
        trained.append({
            "gen_id": gen_id,
            "config_path": config_path,
            "algorithm_path": algorithm_path,
            "model_output_path": model_output_path,
        })

    class FakeJudge:
        def evaluate(
            self,
            *,
            solution_a,
            solution_b,
            games,
            id_a=None,
            id_b=None,
        ):
            assert solution_a.endswith("model_1.pt")
            assert solution_b.endswith("model_2.pt")
            assert games == 3

            return {
                "winner": solution_a,
                "winner_id": id_a,
                "metrics": {
                    "A": {"score": 0.8},
                    "B": {"score": 0.4},
                },
                "matches": [],
            }

    adapter = BrainstormingEvaluationAdapter(
        judge=FakeJudge(),
        train_function=fake_train_function,
    )

    result = adapter.evaluate(
        proposal_1="def get_epsilon(episode_index, hyperparameters): return 0.1",
        proposal_2="def get_epsilon(episode_index, hyperparameters): return 0.2",
        gen_id=1,
        config_1={"epsilon_start": 0.1},
        config_2={"epsilon_start": 0.2},
        games=3,
    )

    assert len(trained) == 2

    assert trained[0]["gen_id"] == 1
    assert trained[1]["gen_id"] == 1

    assert trained[0]["config_path"].endswith("config_1.yaml")
    assert trained[1]["config_path"].endswith("config_2.yaml")

    assert trained[0]["algorithm_path"].endswith("algorithm_1.py")
    assert trained[1]["algorithm_path"].endswith("algorithm_2.py")

    assert trained[0]["model_output_path"].endswith("model_1.pt")
    assert trained[1]["model_output_path"].endswith("model_2.pt")

    assert result["winner"].endswith("model_1.pt")
    assert result["metrics"]["A"]["score"] == 0.8
    assert result["metrics"]["B"]["score"] == 0.4