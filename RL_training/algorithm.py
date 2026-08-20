"""
algorithm.py
------------
RL training logic and hyperparameter defaults for DQLModel.

Maintained public interface:
  - get_hyperparameters
  - get_epsilon
  - select_action
  - play_training_episode
"""

import random

DEFAULT_HYPERPARAMETERS = {
    "gamma": 0.99,                 # Discount factor
    "epsilon_start": 1.0,          # Starting exploration rate
    "epsilon_end": 0.05,           # Minimum exploration rate
    "epsilon_decay_episodes": 500,  # Decay duration in episodes
}


def get_hyperparameters(config: dict) -> dict:
    """Override defaults with config.yaml values if present."""
    hyperparameters = dict(DEFAULT_HYPERPARAMETERS)
    for key in DEFAULT_HYPERPARAMETERS:
        if key in config:
            hyperparameters[key] = config[key]
    return hyperparameters


def get_epsilon(episode_index: int, hyperparameters: dict) -> float:
    """Linear epsilon decay schedule."""
    start = hyperparameters["epsilon_start"]
    end = hyperparameters["epsilon_end"]
    decay_episodes = max(1, hyperparameters["epsilon_decay_episodes"])

    progress = min(1.0, episode_index / decay_episodes)
    return start + (end - start) * progress


def select_action(model, observation, epsilon: float, num_actions: int) -> int:
    """Standard epsilon-greedy policy."""
    if random.random() < epsilon:
        return random.randrange(num_actions)

    q_values = model.predict(observation)
    return int(q_values.argmax())


def play_training_episode(
    env,
    model,
    trained_agent: str,
    hyperparameters: dict,
    epsilon: float,
) -> float:
    """
    Runs one episode in PettingZoo and updates the model online.
    Opponents currently pick random actions.
    """
    env.reset()
    gamma = hyperparameters["gamma"]

    total_reward = 0.0
    pending_state = None
    pending_action = None

    for agent in env.agent_iter():
        observation, reward, termination, truncation, _info = env.last()
        done = termination or truncation

        if agent == trained_agent:
            total_reward += reward
            # Update model based on the previous turn's outcome
            if pending_state is not None:
                model.train_step(
                    state=pending_state,
                    action=pending_action,
                    reward=reward,
                    next_state=observation,
                    done=done,
                    gamma=gamma,
                )

        if done:
            action = None
        else:
            num_actions = env.action_space(agent).n
            if agent == trained_agent:
                action = select_action(model, observation, epsilon, num_actions)
                pending_state = observation
                pending_action = action
            else:
                action = random.randrange(num_actions)

        env.step(action)

    return total_reward
