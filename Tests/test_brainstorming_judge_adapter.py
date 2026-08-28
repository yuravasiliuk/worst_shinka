from worst_shinka.brainstorming_system.judge_adapter import (
    BrainstormingJudgeAdapter,
)


def test_brainstorming_judge_adapter_passes_two_proposals_to_training():
    trained = []

    class FakeJudge:
        def evaluate(
            self,
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
                "winner_id": "proposal-1",
                "metrics": {
                    "A": {"score": 0.8},
                    "B": {"score": 0.4},
                },
                "matches": [],
            }

    def fake_train(
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

        with open(model_output_path, "w") as f:
            f.write("fake model")

    adapter = BrainstormingJudgeAdapter(
        judge=FakeJudge(),
        train_function=fake_train,
    )

    result = adapter.evaluate(
        proposal_1="def select_action(): pass",
        proposal_2="def select_action(): return 1",
        gen_id=5,
        config_1={"gamma": 0.99},
        config_2={"gamma": 0.95, "learning_rate": 0.01},
        games=3,
    )

    assert len(trained) == 2

    assert trained[0]["config_path"].endswith("config_1.yaml")
    assert trained[1]["config_path"].endswith("config_2.yaml")

    assert result["winner_id"] == "proposal-1"
    assert result["metrics"]["A"]["score"] == 0.8
    assert result["metrics"]["B"]["score"] == 0.4