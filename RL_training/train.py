"""
train.py
--------
Training loop for running a single generation. Handles env setup,
model initialization, and logging.

Keep this file static. Pass dynamic logic via config.yaml and algorithm.py.
"""

import importlib.util
import os
import time

import numpy as np
import yaml
from pettingzoo.atari import tennis_v3

from model import DQLModel

# Environment configuration
OBS_TYPE = "ram"            # Atari RAM observation (128 bytes)
INPUT_SIZE = 128             # Input size
TRAINED_AGENT = "first_0"    # Target agent ID in PettingZoo

# Global execution settings
GLOBAL_CONFIG_CANDIDATE_PATHS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "global_config.yaml"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "global_config.yaml"),
]

DEFAULT_GLOBAL_CONFIG = {
    "episodes_per_generation": 100,
    "max_seconds_per_match": 60,
    "max_cycles_per_match": 10000,
    "log_every_n_episodes": 100,
}


def _load_global_config() -> dict:
    """Load system config or fall back to defaults."""
    for path in GLOBAL_CONFIG_CANDIDATE_PATHS:
        if os.path.isfile(path):
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            merged = dict(DEFAULT_GLOBAL_CONFIG)
            merged.update(data)
            return merged
    return dict(DEFAULT_GLOBAL_CONFIG)


def _load_config(config_path: str) -> dict:
    """Load model and algorithm settings from config.yaml."""
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def _load_algorithm(algorithm_path: str):
    """Dynamically load algorithm.py from a given file path."""
    spec = importlib.util.spec_from_file_location("algorithm", algorithm_path)
    algorithm_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(algorithm_module)
    return algorithm_module


def _write_training_logs(log_path: str, episode_rewards: list, log_every: int) -> None:
    """Write interval average rewards to training_logs.txt."""
    rewards = np.array(episode_rewards, dtype=np.float32)

    if len(rewards) == 0:
        averages = np.array([], dtype=np.float32)
    else:
        num_chunks = int(np.ceil(len(rewards) / log_every))
        averages = np.array(
            [
                rewards[i * log_every:(i + 1) * log_every].mean()
                for i in range(num_chunks)
            ],
            dtype=np.float32,
        )

    with open(log_path, "w") as f:
        f.write(f"# Average reward every {log_every} episodes\n")
        for avg in averages:
            f.write(f"{avg}\n")


def train(config_path: str, algorithm_path: str, model_output_path: str) -> None:
    """Train a model end-to-end and save outputs."""
    config = _load_config(config_path)
    algorithm = _load_algorithm(algorithm_path)
    global_config = _load_global_config()

    env = tennis_v3.env(
        render_mode=None,
        obs_type=OBS_TYPE,
        max_cycles=global_config["max_cycles_per_match"],
    )
    num_actions = env.action_space(TRAINED_AGENT).n

    hidden_layers = config.get("hidden_layers", [])
    model = DQLModel(
        input_size=INPUT_SIZE,
        output_size=num_actions,
        hidden_layers=hidden_layers,
    )

    hyperparameters = algorithm.get_hyperparameters(config)

    episodes = global_config["episodes_per_generation"]
    max_seconds_per_match = global_config["max_seconds_per_match"]
    log_every = global_config["log_every_n_episodes"]

    episode_rewards = []
    for episode_index in range(episodes):
        epsilon = algorithm.get_epsilon(episode_index, hyperparameters)

        start_time = time.time()
        total_reward = algorithm.play_training_episode(
            env=env,
            model=model,
            trained_agent=TRAINED_AGENT,
            hyperparameters=hyperparameters,
            epsilon=epsilon,
        )
        elapsed = time.time() - start_time

        if elapsed > max_seconds_per_match:
            print(
                f"[train] Warning: episode {episode_index} took {elapsed:.1f}s, "
                f"exceeding the {max_seconds_per_match}s per-match budget."
            )

        episode_rewards.append(total_reward)

    env.close()

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    model.save(model_output_path)

    log_path = os.path.join(os.path.dirname(model_output_path), "training_logs.txt")
    _write_training_logs(log_path, episode_rewards, log_every)

    print(
        f"[train] Done. Trained {episodes} episodes, "
        f"final avg reward (last {min(log_every, episodes)} eps): "
        f"{np.mean(episode_rewards[-log_every:]):.3f}. "
        f"Model saved to {model_output_path}."
    )


if __name__ == "__main__":
    train(
        config_path="config.yaml",
        algorithm_path="algorithm.py",
        model_output_path=os.path.join("results", "gen_0", "model.pt"),
    )
