import csv
import json
import logging
import os
import time
from itertools import combinations

import torch
import yaml
from pettingzoo.atari import tennis_v3

from utils import (
    ELO_BASELINE,
    MAX_CYCLES,
    PURPLE,
    RAM_GAMES,
    RESULTS_DIR,
    TOURNAMENT_TABLE_PATH,
    _format_duration,
    _load_model,
    _load_model_score_history,
    _model_path,
    _save_model_score_history,
    progress_done,
    progress_line,
)

logger = logging.getLogger(__name__)
ELO_K = 32
STANDINGS_FILE = "tournament_results.csv"
GLOBAL_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "global_config.yaml")
with open(GLOBAL_CONFIG_PATH, encoding="utf-8") as _f:
    _global_config = yaml.safe_load(_f)
MATCHES_PER_PAIR = int(_global_config["scoring"]["matches_per_evaluation"])



def _select_action(observation, model):
    with torch.no_grad():
        obs = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        return int(torch.argmax(model(obs), dim=-1).item())


def _play_match(model_a, model_b):
    env = tennis_v3.env(render_mode=None, obs_type="ram", max_cycles=MAX_CYCLES)
    env.reset()
    ale = env.unwrapped.ale
    models = {"first_0": model_a, "second_0": model_b}
    try:
        for agent in env.agent_iter():
            observation, _reward, termination, truncation, _info = env.last()
            action = None if termination or truncation else _select_action(observation, models[agent])
            env.step(action)
        ram = ale.getRAM()
        return int(ram[RAM_GAMES["first_0"]]), int(ram[RAM_GAMES["second_0"]])
    finally:
        env.close()




def _play_pair_task(gen_a, gen_b, matches_per_pair: int = MATCHES_PER_PAIR):
    """Play the configured number of matches, alternating seat order each round."""
    model_a, model_b = _load_model(_model_path(gen_a)), _load_model(_model_path(gen_b))
    legs = []
    for match_index in range(matches_per_pair):
        if match_index % 2 == 0:
            a_first, b_second = _play_match(model_a, model_b)
            legs.append((a_first, b_second))
        else:
            b_first, a_second = _play_match(model_b, model_a)
            legs.append((a_second, b_first))
    return gen_a, gen_b, legs




def _available_generations():
    result = []
    for name in os.listdir(RESULTS_DIR):
        if not name.startswith("gen_"):
            continue
        try:
            gen_id = int(name.removeprefix("gen_"))
        except ValueError:
            continue
        if os.path.isfile(os.path.join(RESULTS_DIR, name, "model.pt")):
            result.append(gen_id)
    return sorted(result)




def _outcome(score_a, score_b):
    return 1.0 if score_a > score_b else 0.0 if score_a < score_b else 0.5




def _calculate_standings(generations, pair_results):
    standings = {g: {"wins": 0, "losses": 0, "draws": 0, "matches": 0} for g in generations}
    ratings = {g: float(ELO_BASELINE) for g in generations}
    for gen_a, gen_b, legs in sorted(pair_results):
        for score_a, score_b in legs:
            actual_a = _outcome(score_a, score_b)
            expected_a = 1 / (1 + 10 ** ((ratings[gen_b] - ratings[gen_a]) / 400))
            delta = ELO_K * (actual_a - expected_a)
            ratings[gen_a] += delta
            ratings[gen_b] -= delta
            standings[gen_a]["matches"] += 1
            standings[gen_b]["matches"] += 1
            if actual_a == 1:
                standings[gen_a]["wins"] += 1
                standings[gen_b]["losses"] += 1
            elif actual_a == 0:
                standings[gen_a]["losses"] += 1
                standings[gen_b]["wins"] += 1
            else:
                standings[gen_a]["draws"] += 1
                standings[gen_b]["draws"] += 1


    for gen_id, row in standings.items():
        matches = row["matches"]
        points = row["wins"] + 0.5 * row["draws"]
        row["win_rate"] = points / matches if matches else 0.5
        # Smoothed tournament strength: positive, unbounded, and comparable.
        row["score"] = (points + 1.0) / (row["losses"] + 0.5 * row["draws"] + 1.0)
        row["elo"] = ratings[gen_id]
    return standings




def _save_table(generations, pair_results):
    size = max(generations, default=0) + 1
    table = [[None] * size for _ in range(size)]
    for gen_a, gen_b, legs in pair_results:
        table[gen_a][gen_b] = sum(a for a, _ in legs)
        table[gen_b][gen_a] = sum(b for _, b in legs)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(TOURNAMENT_TABLE_PATH, "w", newline="") as handle:
        for row in table:
            handle.write(";".join("None" if value is None else str(value) for value in row) + "\n")




def _save_standings(standings):
    path = os.path.join(RESULTS_DIR, STANDINGS_FILE)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["generation", "wins", "losses", "draws", "matches", "win_rate", "score", "elo"])
        for gen_id in sorted(standings):
            row = standings[gen_id]
            writer.writerow([gen_id, row["wins"], row["losses"], row["draws"], row["matches"], row["win_rate"], row["score"], row["elo"]])




def _update_generation_metrics(standings):
    history = {row[0]: row for row in _load_model_score_history()}
    for gen_id, standing in standings.items():
        row = history.setdefault(gen_id, [gen_id, None, None, None, None])
        row[1], row[3] = standing["elo"], standing["score"]
        metrics_path = os.path.join(RESULTS_DIR, f"gen_{gen_id}", "metrics.json")
        try:
            with open(metrics_path, encoding="utf-8") as handle:
                metrics = json.load(handle)
        except (OSError, json.JSONDecodeError):
            metrics = {"generation": gen_id, "status": "correct"}
        metrics.update(standing)
        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    _save_model_score_history([history[g] for g in sorted(history)])




def run_tournament(gen_id, workers=1):
    generations = _available_generations()
    if gen_id not in generations:
        raise FileNotFoundError(f"Model for generation {gen_id} does not exist")
    pairs = list(combinations(generations, 2))
    if not pairs:
        standings = _calculate_standings(generations, [])
        _save_table(generations, [])
        _save_standings(standings)
        _update_generation_metrics(standings)
        logger.info("[gen 0] tournament standings initialized")
        return False


    logger.info("[gen %s] full round-robin — %s models, %s pairs, %s matches per pair", gen_id, len(generations), len(pairs), MATCHES_PER_PAIR)
    started = time.time()
    if workers > 1:
        import multiprocessing
        context = multiprocessing.get_context("spawn")
        with context.Pool(processes=min(workers, len(pairs))) as pool:
            pair_results = pool.starmap(_play_pair_task, [(a, b, MATCHES_PER_PAIR) for a, b in pairs])
    else:
        pair_results = [_play_pair_task(a, b, MATCHES_PER_PAIR) for a, b in pairs]


    for index, (a, b, legs) in enumerate(pair_results, 1):
        progress_line("tournament", label_color=PURPLE, generation=gen_id,
                      progress=f"pair {index}/{len(pairs)}: gen_{a} vs gen_{b}",
                      result=f"{legs[0][0]}-{legs[0][1]}, {legs[1][0]}-{legs[1][1]}")
    progress_done()
    standings = _calculate_standings(generations, pair_results)
    _save_table(generations, pair_results)
    _save_standings(standings)
    _update_generation_metrics(standings)
    current = standings[gen_id]
    logger.info("🏆 [gen %s] tournament complete in %s | W-L-D %s-%s-%s | score %.3f | elo %.1f",
                gen_id, _format_duration(time.time() - started), current["wins"], current["losses"],
                current["draws"], current["score"], current["elo"])
    return True
