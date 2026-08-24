"""
algorithm.py
------------
RL policy logic for DQLModel. This is the only file an LLM may change
between generations — no environment/loop mechanics and no config
defaults here, only decision logic. Hyperparameters always come from
the generation's config.yaml, loaded and passed in by train.py.

Maintained public interface:
  - get_epsilon
  - select_action
  - select_opponent_action
  - update_model
"""

import random


def get_epsilon(episode_index: int, hyperparameters: dict) -> float:
    """Linear epsilon decay schedule."""
    start = hyperparameters["epsilon_start"]
    end = hyperparameters["epsilon_end"]
    decay_episodes = max(1, hyperparameters["epsilon_decay_episodes"])

    progress = min(1.0, episode_index / decay_episodes)
    return start + (end - start) * progress


def select_action(model, observation, epsilon: float, num_actions: int) -> int:
    """Standard epsilon-greedy policy for the trained agent."""
    if random.random() < epsilon:
        return random.randrange(num_actions)

    q_values = model.predict(observation)
    return int(q_values.argmax())


def select_opponent_action(model, observation, num_actions: int) -> int:
    """Opponent behavior during training: plays greedily off a past-generation
    model if one was given, otherwise a random action (gen0 has no past
    generations to draw from)."""
    if model is None:
        return random.randrange(num_actions)

    q_values = model.predict(observation)
    return int(q_values.argmax())


def update_model(model, state, action, reward, next_state, done, hyperparameters) -> None:
    """Applies one online training update to the model."""
    model.train_step(
        state=state,
        action=action,
        reward=reward,
        next_state=next_state,
        done=done,
        gamma=hyperparameters["gamma"],
    )
