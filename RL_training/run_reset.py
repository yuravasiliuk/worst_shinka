"""
run_reset.py

Resets training progress: deletes every generation folder except gen_0,
deletes gen_0's generated training artifacts (model.pt, training_logs.txt)
while keeping its config.yaml/algorithm.py, and clears the shared
tournament_table.csv and model_score_history.csv.
"""

import os
import shutil

from utils import RESULTS_DIR

GEN0_ARTIFACTS = ["model.pt", "training_logs.txt"]
SHARED_RESULT_FILES = ["tournament_table.csv", "model_score_history.csv"]


def run_reset() -> None:
    if not os.path.isdir(RESULTS_DIR):
        return

    for entry in os.listdir(RESULTS_DIR):
        entry_path = os.path.join(RESULTS_DIR, entry)

        if entry == "gen_0":
            for artifact in GEN0_ARTIFACTS:
                artifact_path = os.path.join(entry_path, artifact)
                if os.path.isfile(artifact_path):
                    os.remove(artifact_path)
        elif os.path.isdir(entry_path):
            shutil.rmtree(entry_path)

    for filename in SHARED_RESULT_FILES:
        file_path = os.path.join(RESULTS_DIR, filename)
        open(file_path, "w").close()


if __name__ == "__main__":
    run_reset()
