from RL_training.brainstorming_adapter import (
    validate_algorithm_code,
    save_algorithm,
)


VALID_ALGORITHM = """
def get_epsilon(episode_index, hyperparameters):
    return 0.1


def select_action(model, observation, epsilon, num_actions):
    return 0


def select_opponent_action(model, observation, num_actions):
    return 0


def update_model(model, state, action, reward, next_state, done, hyperparameters):
    pass
"""


INVALID_ALGORITHM = """
def get_epsilon(episode_index, hyperparameters):
    return 0.1
"""


def test_valid_algorithm():
    valid, reason = validate_algorithm_code(VALID_ALGORITHM)

    assert valid is True
    assert reason == "OK"


def test_invalid_algorithm_missing_functions():
    valid, reason = validate_algorithm_code(INVALID_ALGORITHM)

    assert valid is False
    assert "Missing required functions" in reason


def test_invalid_python():
    code = """
def get_epsilon(
"""

    valid, reason = validate_algorithm_code(code)

    assert valid is False
    assert "SyntaxError" in reason


def test_save_algorithm(tmp_path):
    output = tmp_path / "algorithm.py"

    result = save_algorithm(
        VALID_ALGORITHM,
        output,
    )

    assert result == output
    assert output.exists()
    assert output.read_text(encoding="utf-8") == VALID_ALGORITHM