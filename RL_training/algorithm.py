"""
algorithm.py
------------
Strategy module for training the DQLModel.

This module houses all tunable hyperparameters and RL logic:
  - Default training hyperparameters
  - Exploration strategy (epsilon decay)
  - Action selection policy
  - Single-episode training loop (online DQN updates)

Note: train.py relies on the public interface (get_hyperparameters,
get_epsilon, and play_training_episode). Keep these function signatures
intact when tweaking internal implementation details.
"""

import random

# Default fallback values. Any key defined in config.yaml will override
# these settings at runtime.
DEFAULT_HYPERPARAMETERS = {
    "gamma": 0.99,                 # Discount factor for future rewards
    "epsilon_start": 1.0,          # Initial exploration rate
    "epsilon_end": 0.05,           # Minimum exploration floor
    "epsilon_decay_episodes": 500,  # Linear decay schedule duration
}


def get_hyperparameters(config: dict) -> dict:
    """Merges config settings with system defaults, favoring user overrides."""
    hyperparameters = dict(DEFAULT_HYPERPARAMETERS)
    for key in DEFAULT_HYPERPARAMETERS:
        if key in config:
            hyperparameters[key] = config[key]
    return hyperparameters


def get_epsilon(episode_index: int, hyperparameters: dict) -> float:
    """Calculates linear epsilon decay based on current episode progress."""
    start = hyperparameters["epsilon_start"]
    end = hyperparameters["epsilon_end"]
    decay_episodes = max(1, hyperparameters["epsilon_decay_episodes"])

    progress = min(1.0, episode_index / decay_episodes)
    return start + (end - start) * progress


def select_action(model, observation, epsilon: float, num_actions: int) -> int:
    """Epsilon-greedy action selection."""
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
    Runs a single training episode in the PettingZoo AEC environment.

    Performs online 1-step Q-learning updates whenever `trained_agent` takes
    a turn. Opponents currently execute random actions as a baseline.

    Returns:
        float: Cumulative reward earned by `trained_agent` during the match.
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
            # Update Q-values using the outcome (reward + new state) of our previous action
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
                # Dummy opponent behavior
                action = random.randrange(num_actions)

        env.step(action)

    return total_reward