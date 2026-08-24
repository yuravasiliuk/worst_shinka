from worst_shinka.cli import integrations


def test_judge_candidates_selects_winner(monkeypatch):
    candidates = [
        {
            "id": "candidate-A",
            "model_path": "model_A.pt",
        },
        {
            "id": "candidate-B",
            "model_path": "model_B.pt",
        },
    ]

    class FakeJudge:
        def evaluate(self, solution_a, solution_b, games, id_a, id_b):
            assert solution_a == "model_A.pt"
            assert solution_b == "model_B.pt"
            assert id_a == "candidate-A"
            assert id_b == "candidate-B"

            return {
                "winner": solution_a,
                "winner_id": id_a,
                "metrics": {
                    "A": {
                        "score": 0.8,
                    },
                    "B": {
                        "score": 0.4,
                    },
                },
                "matches": [],
            }

    monkeypatch.setattr(integrations, "Judge", FakeJudge)

    result = integrations.judge_candidates(
        candidates=candidates,
    )

    assert len(result) == 1
    assert result[0]["id"] == "candidate-A"
    assert result[0]["model_path"] == "model_A.pt"
    assert result[0]["score"] == 0.8