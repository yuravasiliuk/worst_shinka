import importlib.util
import logging
import os
import random
import time
import torch
import numpy as np
import yaml
from pettingzoo.atari import tennis_v3

from model import DQLModel
from utils import (
    ELO_BASELINE,
    GREEN,
    MAX_CYCLES,
    _format_duration,
    _load_model,
    _load_model_score_history,
    _model_path,
    _save_model_score_history,
    progress_done,
    progress_line,
)

logger = logging.getLogger(__name__)

# Environment configuration
OBS_TYPE = "ram"            # Atari RAM observation (128 bytes)
INPUT_SIZE = 128             # Input size
TRAINED_AGENT = "first_0"    # Target agent ID in PettingZoo

# How often (in episodes) to refresh the in-place training progress line.
PROGRESS_LOG_EVERY = 10

# Global execution settings
GLOBAL_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "global_config.yaml")


def _load_global_config() -> dict:
    with open(GLOBAL_CONFIG_PATH) as f:
        return yaml.safe_load(f)["training"]


def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def _load_algorithm(algorithm_path: str):
    spec = importlib.util.spec_from_file_location("algorithm", algorithm_path)
    algorithm_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(algorithm_module)
    return algorithm_module


def _play_training_episode(env, model, opponent_model, algorithm, epsilon: float, config: dict) -> float:
    env.reset()

    total_reward = 0.0
    pending_state = None
    pending_action = None

    for agent in env.agent_iter():
        observation, reward, termination, truncation, _info = env.last()
        done = termination or truncation

        if agent == TRAINED_AGENT:
            total_reward += reward
            if pending_state is not None:
                algorithm.update_model(
                    model=model,
                    state=pending_state,
                    action=pending_action,
                    reward=reward,
                    next_state=observation,
                    done=done,
                    hyperparameters=config,
                )

        if done:
            action = None
        elif agent == TRAINED_AGENT:
            num_actions = env.action_space(agent).n
            action = algorithm.select_action(model, observation, epsilon, num_actions)
            pending_state = observation
            pending_action = action
        else:
            num_actions = env.action_space(agent).n
            action = algorithm.select_opponent_action(opponent_model, observation, num_actions)

        env.step(action)

    return total_reward


def _write_training_logs(log_path: str, episode_rewards: list, log_every: int) -> None:
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
        f.write(";".join(f"{avg:.2f}" for avg in averages) + "\n")


def train(gen_id: int, config_path: str, algorithm_path: str, model_output_path: str) -> None:
    """Train a model end-to-end and save outputs."""
    config = _load_config(config_path)
    algorithm = _load_algorithm(algorithm_path)
    global_config = _load_global_config()

    env = tennis_v3.env(
        render_mode=None,
        obs_type=OBS_TYPE,
        max_cycles=MAX_CYCLES,
    )
    num_actions = env.action_space(TRAINED_AGENT).n

    hidden_layers = config.get("hidden_layers", [])

    model = DQLModel(
        input_size=INPUT_SIZE,
        output_size=num_actions,
        hidden_layers=hidden_layers,
    )
    logger.info("[gen %s] model device: %s", gen_id, model.device)

    # Train against a random past generation (non-transitive matchups are
    # less of a risk than always training against just gen_id - 1). gen0
    # has no past generations, so it trains against random actions.
    if gen_id == 0:
        opponent_model = None
        opponent_label = "random actions (gen_0 has no history)"
    else:
        opponent_gen_id = random.randrange(gen_id)
        opponent_model = _load_model(_model_path(opponent_gen_id))
        opponent_label = f"gen_{opponent_gen_id} model"

    episodes = global_config["episodes_per_model"]
    max_seconds_per_match = global_config["max_seconds_per_match"]
    log_every = global_config["log_every_n_episodes"]

    logger.info("[gen %s] starting training — %s episodes vs %s", gen_id, episodes, opponent_label)

    episode_rewards = []
    best_reward = float("-inf")
    train_start = time.time()
    for episode_index in range(episodes):
        epsilon = algorithm.get_epsilon(episode_index, config)

        start_time = time.time()
        total_reward = _play_training_episode(env, model, opponent_model, algorithm, epsilon, config)
        elapsed = time.time() - start_time

        if elapsed > max_seconds_per_match:
            logger.warning(
                "[gen %s] episode %s took %.1fs, exceeding the %ss per-match budget.",
                gen_id, episode_index, elapsed, max_seconds_per_match,
            )

        episode_rewards.append(total_reward)
        best_reward = max(best_reward, total_reward)

        completed = episode_index + 1
        if completed % PROGRESS_LOG_EVERY == 0 or completed == episodes:
            window = episode_rewards[-PROGRESS_LOG_EVERY:]
            avg_reward = sum(window) / len(window)
            total_elapsed = time.time() - train_start
            speed = completed / total_elapsed if total_elapsed > 0 else 0.0
            eta_seconds = (episodes - completed) / speed if speed > 0 else 0.0
            progress_line(
                "training",
                label_color=GREEN,
                generation=gen_id,
                progress=f"{completed}/{episodes} episodes ({completed / episodes:.0%})",
                result=(
                    f"avg reward {avg_reward:.2f} (best {best_reward:.2f}), "
                    f"ε={epsilon:.2f}, {speed:.1f} ep/s, ETA {_format_duration(eta_seconds)}"
                ),
            )

    progress_done()
    training_duration = time.time() - train_start
    env.close()

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    model.save(model_output_path)

    log_path = os.path.join(os.path.dirname(model_output_path), "training_logs.txt")
    _write_training_logs(log_path, episode_rewards, log_every)

    average_score = episode_rewards[-1] if episode_rewards else None
    initial_elo = ELO_BASELINE if gen_id == 0 else None
    if not os.environ.get("WORST_SHINKA_SKIP_HISTORY"):
        score_history = _load_model_score_history()
        score_history.append([gen_id, initial_elo, average_score, round(training_duration, 2)])
        _save_model_score_history(score_history)

    logger.info(
        "[gen %s] training complete — %s episodes in %s | avg reward (last %s): %.3f | best: %.2f | saved to %s",
        gen_id,
        episodes,
        _format_duration(training_duration),
        min(log_every, episodes),
        np.mean(episode_rewards[-log_every:]),
        best_reward,
        model_output_path,
    )


if __name__ == "__main__":
    train(
        gen_id=0,
        config_path=os.path.join("results", "gen_0", "config.yaml"),
        algorithm_path=os.path.join("results", "gen_0", "algorithm.py"),
        model_output_path=os.path.join("results", "gen_0", "model.pt"),
    )
