import logging
import os
import time

import torch
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


def _tournament_match_task(current_model_path, opponent_model_path):
    current_model = _load_model(current_model_path)
    opponent_model = _load_model(opponent_model_path)
    return _play_match(current_model, opponent_model)


def _select_action(observation, model):
    with torch.no_grad():
        obs = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        logits = model(obs)
        return int(torch.argmax(logits, dim=-1).item())


def _play_match(model_a, model_b):
    # model_a plays first_0, model_b plays second_0. No rendering.
    env = tennis_v3.env(render_mode=None, obs_type="ram", max_cycles=MAX_CYCLES)
    env.reset()
    ale = env.unwrapped.ale
    models = {"first_0": model_a, "second_0": model_b}

    for agent in env.agent_iter():
        observation, _reward, termination, truncation, _info = env.last()
        action = None if (termination or truncation) else _select_action(observation, models[agent])
        env.step(action)

    ram = ale.getRAM()
    games_a = int(ram[RAM_GAMES["first_0"]])
    games_b = int(ram[RAM_GAMES["second_0"]])
    env.close()
    return games_a, games_b


def _load_table():
    if not os.path.isfile(TOURNAMENT_TABLE_PATH):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        open(TOURNAMENT_TABLE_PATH, "w").close()
        return []
    with open(TOURNAMENT_TABLE_PATH) as f:
        rows = [line.strip().split(";") for line in f if line.strip()]
    return [[None if v == "None" else int(v) for v in row] for row in rows]


def _save_table(table):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(TOURNAMENT_TABLE_PATH, "w") as f:
        for row in table:
            f.write(";".join("None" if v is None else str(v) for v in row) + "\n")


def run_tournament(gen_id, workers=1):
    if (gen_id == 0):
        _save_table([[None]])
        logger.info("[gen 0] tournament skipped — no prior generations to play against")
        return False


    current_model = _load_model(_model_path(gen_id))
    opponent_models = {i: _load_model(_model_path(i)) for i in range(gen_id)}

    # Expanding table
    table = _load_table()
    for row in table:
        row.append(None)
    table.append([None] * (gen_id + 1))

    # Playing tournament matches
    score_history = _load_model_score_history()
    opponent_elo = {row[0]: (ELO_BASELINE if row[1] is None else row[1]) for row in score_history}

    total_matches = len(opponent_models)
    logger.info("[gen %s] starting tournament — %s match(es) vs prior generations", gen_id, total_matches)
    tournament_start = time.time()

    if workers > 1:
        import multiprocessing

        context = multiprocessing.get_context("spawn")
        tasks = [(_model_path(gen_id), _model_path(opp_id)) for opp_id in opponent_models]
        with context.Pool(processes=min(workers, total_matches)) as pool:
            match_results = pool.starmap(_tournament_match_task, tasks)
    else:
        match_results = [
            _play_match(current_model, opponent_model)
            for opponent_model in opponent_models.values()
        ]

    actual_total = 0.0
    expected_total = 0.0
    for match_index, ((opp_id, _opponent_model), (games_current, games_opp)) in enumerate(
        zip(opponent_models.items(), match_results), start=1
    ):
        table[gen_id][opp_id] = games_current
        table[opp_id][gen_id] = games_opp

        if games_current > games_opp:
            actual = 1.0
            outcome = "win"
        elif games_current < games_opp:
            actual = 0.0
            outcome = "loss"
        else:
            actual = 0.5
            outcome = "draw"
        expected = 1 / (1 + 10 ** ((opponent_elo.get(opp_id, ELO_BASELINE) - ELO_BASELINE) / 400))
        actual_total += actual
        expected_total += expected

        progress_line(
            "tournament",
            label_color=PURPLE,
            generation=gen_id,
            progress=f"match {match_index}/{total_matches} vs gen_{opp_id}",
            result=f"{games_current}-{games_opp} ({outcome}), elapsed {_format_duration(time.time() - tournament_start)}",
        )

    progress_done()
    _save_table(table)

    # Batch ELO update: one aggregate update for gen_id vs. the whole round,
    # not one update per match.
    num_matches = total_matches
    new_elo = ELO_BASELINE + ELO_K * (actual_total / num_matches - expected_total / num_matches)
    for row in score_history:
        if row[0] == gen_id:
            row[1] = new_elo
            break
    _save_model_score_history(score_history)

    logger.info(
        "🏆 [gen %s] tournament complete — %s match(es) in %s | elo %.0f (%+.0f vs baseline %s)",
        gen_id, num_matches, _format_duration(time.time() - tournament_start), new_elo, new_elo - ELO_BASELINE, ELO_BASELINE,
    )

    return True


