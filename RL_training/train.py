"""
train.py
--------
Core training harness for executing single-generation DQL runs.

This module serves as the fixed infrastructure layer across experiments:
  - Initializes the PettingZoo Atari Tennis environment (RAM observations)
  - Instantiates the DQLModel using parameters defined in config.yaml
  - Executes the training loop via the dynamic algorithm module
  - Records aggregated training metrics (training_logs.txt)
  - Exports the trained network weights (model.pt)

Note: This file represents core infrastructure and should remain constant.
Strategy variations are passed dynamically via config.yaml and algorithm.py.

Entry point (called by run_training.py):
    train(config_path=..., algorithm_path=..., model_output_path=...)
"""

import importlib.util
import os
import time

import numpy as np
import yaml
from pettingzoo.atari import tennis_v3

from model import DQLModel

# --- Static environment configuration ---
OBS_TYPE = "ram"            # Atari RAM observation space (128-byte vector)
INPUT_SIZE = 128             # Input state dimension
TRAINED_AGENT = "first_0"    # Primary agent identifier in PettingZoo

# --- Global training settings ---
# Loads top-level defaults if a global_config.yaml isn't present in parent directories.
GLOBAL_CONFIG_CANDIDATE_PATHS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "global_config.yaml"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "global_config.yaml"),
]

DEFAULT_GLOBAL_CONFIG = {
    "episodes_per_generation": 100,   # Total matches per training run
    "max_seconds_per_match": 60,      # Match duration threshold warning
    "max_cycles_per_match": 10000,    # Hard limit on env steps per match
    "log_every_n_episodes": 100,      # Logging aggregation window
}


def _load_global_config() -> dict:
    """Loads system-wide execution parameters with fallback to defaults."""
    for path in GLOBAL_CONFIG_CANDIDATE_PATHS:
        if os.path.isfile(path):
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            merged = dict(DEFAULT_GLOBAL_CONFIG)
            merged.update(data)
            return merged
    return dict(DEFAULT_GLOBAL_CONFIG)


def _load_config(config_path: str) -> dict:
    """Loads specific model architecture and strategy configs."""
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def _load_algorithm(algorithm_path: str):
    """
    Dynamically imports algorithm.py from a specified file path,
    allowing variable strategy modules to be loaded per experiment.
    """
    spec = importlib.util.spec_from_file_location("algorithm", algorithm_path)
    algorithm_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(algorithm_module)
    return algorithm_module


def _write_training_logs(log_path: str, episode_rewards: list, log_every: int) -> None:
    """
    Computes moving window averages of episode rewards and writes them to a log file.
    """
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
    """
    Executes an end-to-end training pipeline.

    Args:
        config_path: Path to config.yaml specifying model architecture 
            and training hyperparameter overrides.
        algorithm_path: Path to the target algorithm.py implementation module.
        model_output_path: Output file path for the saved PyTorch model checkpoint.
    """
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
    # Local integration test entry point
    train(
        config_path="config.yaml",
        algorithm_path="algorithm.py",
        model_output_path=os.path.join("results", "gen_0", "model.pt"),
    )