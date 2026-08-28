from worst_shinka.judge import Judge, JudgeConfig


def test_judge_is_in_separate_package():
    judge = Judge()
    config = JudgeConfig()

    assert isinstance(judge, Judge)
    assert config.win_rate_weight == 0.60
    assert config.game_difference_weight == 0.25
    assert config.consistency_weight == 0.15