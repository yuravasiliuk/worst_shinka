"""
algorithm.py
------------
RL policy logic for DQLModel. This gen_0 copy is the fixed baseline/greedy
reference seed — it and its sibling config.yaml are never LLM-generated.
From generation 1 onward, brainstorming LLMs may propose changes to both
this file's decision logic and to config.yaml's hyperparameters (values
and/or new keys) together, as a paired proposal — no environment/loop
mechanics belong here, only decision logic. Hyperparameters always come
from the generation's own config.yaml, loaded and passed in by train.py.

Maintained public interface:
  - get_epsilon
  - select_action
  - select_opponent_action
  - update_model

This gen_0 baseline's config.yaml declares these hyperparameters (later
generations may retune these values and/or introduce new keys):
- hidden_layers
- gamma
- epsilon_start
- epsilon_end
- epsilon_decay_episodes
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
