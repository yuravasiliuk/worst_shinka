import os
import torch
import yaml

GLOBAL_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "global_config.yaml")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
TOURNAMENT_TABLE_PATH = os.path.join(RESULTS_DIR, "tournament_table.csv")
RAM_POINTS = {"first_0": 69, "second_0": 70}
RAM_GAMES = {"first_0": 71, "second_0": 72}
MAX_CYCLES = 1000

with open(GLOBAL_CONFIG_PATH) as _f:
    MAX_CYCLES = yaml.safe_load(_f)["training"]["max_steps_per_episode"]


def _model_path(gen_id):
    path = os.path.join(RESULTS_DIR, f"gen{gen_id}", "model.pt")
    if not os.path.isfile(path):
        raise Exception(f"Path to model {gen_id} doesn't exist")
    return path

def _load_model(path):
    model = torch.load(path, weights_only=False, map_location="cpu")
    model.eval()
    return model
