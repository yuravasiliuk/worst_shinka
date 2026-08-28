from unittest.mock import Mock

from worst_shinka.brainstorming_system.brainstorm import (
    BrainstormResult,
    EvolutionWorkflow,
)


VALID_PROPOSAL = """
def get_epsilon(episode_index, hyperparameters):
    return 0.0


def select_action(model, observation, epsilon, num_actions):
    return 0


def select_opponent_action(model, observation, num_actions):
    return 0


def update_model(model, state, action, reward, next_state, done, hyperparameters):
    model.train_step(state, action, reward, next_state, done, 0.0)
"""


def test_failed_judge_retries_three_times_and_returns_control():
    workflow = EvolutionWorkflow.__new__(EvolutionWorkflow)
    workflow.brainstormer = Mock()
    workflow.judge = Mock()
    workflow.hyperparameter_keys = []

    workflow.brainstormer.run_brainstorming.side_effect = [
        BrainstormResult(VALID_PROPOSAL, VALID_PROPOSAL, [])
        for _ in range(3)
    ]
    workflow.judge.evaluate.return_value = {
        "winner": None,
        "metrics": {"A": {}, "B": {}},
    }

    result = workflow.execute_crossover([{}])

    assert result is None
    assert workflow.brainstormer.run_brainstorming.call_count == 3
    assert workflow.judge.evaluate.call_count == 3
