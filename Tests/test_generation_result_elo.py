from worst_shinka.cli.orchestrator import _generation_result_row


def test_generation_result_row_keeps_elo_from_lineage():
    row = _generation_result_row(
        {
            "generation": 7,
            "status": "correct",
            "average_training_score": 12.5,
            "score": 0.87,
            "elo": 1342.0,
            "time": 123.5,
        },
        generation=7,
        generation_cost=0.42,
    )

    assert row["elo"] == 1342.0
    assert row["status"] == "correct"
    assert row["score"] == 0.87
    assert row["cost"] == 0.42
