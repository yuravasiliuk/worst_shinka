from worst_shinka.judge import Judge, JudgeConfig


def test_judge_config_default_weights():
    config = JudgeConfig()

    assert config.win_rate_weight == 0.60
    assert config.game_difference_weight == 0.25
    assert config.consistency_weight == 0.15


def test_judge_config_rejects_invalid_weights():
    try:
        JudgeConfig(
            win_rate_weight=0.8,
            game_difference_weight=0.3,
            consistency_weight=0.1,
        )
        assert False, "JudgeConfig should reject weights that do not sum to 1"
    except ValueError:
        pass


def test_difference_score():
    assert Judge._difference_score(0) == 0.5
    assert Judge._difference_score(1) == 0.6
    assert Judge._difference_score(-1) == 0.4

    # Check bounds
    assert Judge._difference_score(100) == 1.0
    assert Judge._difference_score(-100) == 0.0


def test_consistency_score():
    results = [
        {"winner": "A"},
        {"winner": "A"},
        {"winner": "draw"},
        {"winner": "B"},
    ]

    assert Judge._consistency_score(results, "A") == 0.625
    assert Judge._consistency_score(results, "B") == 0.375


def test_consistency_score_empty():
    assert Judge._consistency_score([], "A") == 0.5


def test_judge_evaluate_calculates_metrics(monkeypatch):
    judge = Judge()

    fake_results = [
        {
            "match": 0,
            "a_games": 5,
            "b_games": 3,
            "winner": "A",
        },
        {
            "match": 1,
            "a_games": 4,
            "b_games": 2,
            "winner": "A",
        },
        {
            "match": 2,
            "a_games": 3,
            "b_games": 3,
            "winner": "draw",
        },
        {
            "match": 3,
            "a_games": 2,
            "b_games": 4,
            "winner": "B",
        },
    ]

    def fake_evaluate_matches(solution_a, solution_b, games):
        assert solution_a == "solution_A"
        assert solution_b == "solution_B"
        assert games == 4

        return fake_results

    monkeypatch.setattr(
        judge,
        "_evaluate_matches",
        fake_evaluate_matches,
    )

    result = judge.evaluate(
        solution_a="solution_A",
        solution_b="solution_B",
        games=4,
        id_a="A",
        id_b="B",
    )

    assert result["winner"] == "solution_A"
    assert result["winner_id"] == "A"

    metrics_a = result["metrics"]["A"]
    metrics_b = result["metrics"]["B"]

    # A: 2 wins + 1 draw
    assert metrics_a["win_rate"] == 0.625

    # Average game difference:
    # (5-3 + 4-2 + 3-3 + 2-4) / 4 = 0.5
    assert metrics_a["game_difference"] == 0.5

    # 0.5 + 0.1 * 0.5 = 0.55
    assert metrics_a["difference_score"] == 0.55

    # 2 wins + 0.5 draw / 4
    assert metrics_a["consistency"] == 0.625

    # Final:
    # 0.60 * 0.625
    # + 0.25 * 0.55
    # + 0.15 * 0.625
    expected_score_a = (
        0.60 * 0.625
        + 0.25 * 0.55
        + 0.15 * 0.625
    )

    assert abs(metrics_a["score"] - expected_score_a) < 1e-9

    # B gets the complementary metrics
    assert metrics_b["win_rate"] == 0.375
    assert metrics_b["game_difference"] == -0.5
    assert abs(metrics_b["difference_score"] - 0.45) < 1e-9
    assert metrics_b["consistency"] == 0.375


def test_judge_evaluate_rejects_zero_games():
    judge = Judge()

    try:
        judge.evaluate(
            solution_a="A",
            solution_b="B",
            games=0,
        )
        assert False, "Judge should reject games=0"
    except ValueError:
        pass