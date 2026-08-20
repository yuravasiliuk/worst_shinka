import os
import torch
import yaml

GLOBAL_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "global_config.yaml")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
TOURNAMENT_TABLE_PATH = os.path.join(RESULTS_DIR, "tournament_table.csv")
MODEL_SCORE_HISTORY_PATH = os.path.join(RESULTS_DIR, "model_score_history.csv")
RAM_GAMES = {"first_0": 71, "second_0": 72}
ELO_BASELINE = 1200
MAX_CYCLES = 1000

with open(GLOBAL_CONFIG_PATH) as _f:
    MAX_CYCLES = yaml.safe_load(_f)["training"]["max_steps_per_episode"]


def _model_path(gen_id):
    path = os.path.join(RESULTS_DIR, f"gen_{gen_id}", "model.pt")
    if not os.path.isfile(path):
        raise Exception(f"Path to model {gen_id} doesn't exist")
    return path

def _load_model(path):
    model = torch.load(path, weights_only=False, map_location="cpu")
    model.eval()
    return model


def _load_model_score_history():
    if not os.path.isfile(MODEL_SCORE_HISTORY_PATH):
        return []
    with open(MODEL_SCORE_HISTORY_PATH) as f:
        rows = [line.strip().split(";") for line in f if line.strip()]
    return [[int(r[0]), None if r[1] == "None" else float(r[1]), None if r[2] == "None" else float(r[2])] for r in rows]


def _save_model_score_history(rows):
    with open(MODEL_SCORE_HISTORY_PATH, "w") as f:
        for gen_id, elo, avg in rows:
            f.write(f"{gen_id};{'None' if elo is None else elo};{'None' if avg is None else avg}\n")
