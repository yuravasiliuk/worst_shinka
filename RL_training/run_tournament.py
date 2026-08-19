import os

import torch
from pettingzoo.atari import tennis_v3

from utils import MAX_CYCLES, RAM_GAMES, RESULTS_DIR, TOURNAMENT_TABLE_PATH, _load_model, _model_path

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
    with open(TOURNAMENT_TABLE_PATH) as f:
        rows = [line.strip().split(";") for line in f if line.strip()]
    return [[None if v == "None" else int(v) for v in row] for row in rows]


def _save_table(table):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(TOURNAMENT_TABLE_PATH, "w") as f:
        for row in table:
            f.write(";".join("None" if v is None else str(v) for v in row) + "\n")


def run_tournament(gen_id):
    if (gen_id == 0):
        _save_table([[None]])
        return False

    
    current_model = _load_model(_model_path(gen_id))
    opponent_models = {i: _load_model(_model_path(i)) for i in range(gen_id)}

    # Expanding table
    table = _load_table()
    for row in table:
        row.append(None)
    table.append([None] * (gen_id + 1))

    # Playing tournament matches
    for opp_id, opponent_model in opponent_models.items():
        games_current, games_opp = _play_match(current_model, opponent_model)
        table[gen_id][opp_id] = games_current
        table[opp_id][gen_id] = games_opp

    _save_table(table)
    return True
    

