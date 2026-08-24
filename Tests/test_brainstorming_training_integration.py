from pathlib import Path

from RL_training.brainstorming_adapter import save_algorithm


VALID_ALGORITHM = """
def get_epsilon(episode_index, hyperparameters):
    start = hyperparameters["epsilon_start"]
    end = hyperparameters["epsilon_end"]
    decay = max(1, hyperparameters["epsilon_decay_episodes"])

    progress = min(1.0, episode_index / decay)
    return start + (end - start) * progress


def select_action(model, observation, epsilon, num_actions):
    import random

    if random.random() < epsilon:
        return random.randrange(num_actions)

    q_values = model.predict(observation)
    return int(q_values.argmax())


def select_opponent_action(model, observation, num_actions):
    if model is None:
        import random
        return random.randrange(num_actions)

    q_values = model.predict(observation)
    return int(q_values.argmax())


def update_model(
    model,
    state,
    action,
    reward,
    next_state,
    done,
    hyperparameters,
):
    model.train_step(
        state=state,
        action=action,
        reward=reward,
        next_state=next_state,
        done=done,
        gamma=hyperparameters["gamma"],
    )
"""


def test_brainstorming_proposal_can_become_training_algorithm(tmp_path):
    algorithm_path = tmp_path / "algorithm.py"

    result = save_algorithm(
        VALID_ALGORITHM,
        algorithm_path,
    )

    assert result == algorithm_path
    assert algorithm_path.exists()

    # The file must be importable as a Python module.
    source = algorithm_path.read_text(encoding="utf-8")

    namespace = {}
    exec(compile(source, str(algorithm_path), "exec"), namespace)

    assert callable(namespace["get_epsilon"])
    assert callable(namespace["select_action"])
    assert callable(namespace["select_opponent_action"])
    assert callable(namespace["update_model"])