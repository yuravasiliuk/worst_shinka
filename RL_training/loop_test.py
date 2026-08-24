import os
import shutil

from run_agreggate_data import run_agreggate_data
from run_tournament import run_tournament
from run_training import run_training

NUM_GENERATIONS = 4

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INITIAL_MODEL_DIR = os.path.join(_SCRIPT_DIR, "..", "initial_model")
TEMPLATE_CONFIG = os.path.join(_SCRIPT_DIR, "_loop_test_template_config.yaml")
TEMPLATE_ALGORITHM = os.path.join(_SCRIPT_DIR, "_loop_test_template_algorithm.py")


def _print_aggregate_data_check(data):
    for key in ("config", "algorithm", "training_logs", "model_score_history", "tournament_table"):
        print(f"{key} - {'tak' if data.get(key) is not None else 'nie'}")


def main():
    shutil.copy(os.path.join(INITIAL_MODEL_DIR, "config.yaml"), TEMPLATE_CONFIG)
    shutil.copy(os.path.join(INITIAL_MODEL_DIR, "algorithm.py"), TEMPLATE_ALGORITHM)

    try:
        for gen_id in range(NUM_GENERATIONS):
            print(f"=== Generation {gen_id} ===")
            run_training(gen_id=gen_id, config=TEMPLATE_CONFIG, algorithm=TEMPLATE_ALGORITHM)
            run_tournament(gen_id)
            data = run_agreggate_data(gen_id)
            _print_aggregate_data_check(data)
    finally:
        os.remove(TEMPLATE_CONFIG)
        os.remove(TEMPLATE_ALGORITHM)


if __name__ == "__main__":
    main()
