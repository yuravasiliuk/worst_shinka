import logging
import os

from utils import RESULTS_DIR

logger = logging.getLogger(__name__)
def run_aggregate_data(gen_id: int) -> dict:
    #Path to this generation's folder
    gen_folder = os.path.join(RESULTS_DIR, f"gen_{gen_id}")
    checklist = []

    # config.yaml: returned as raw text, the caller can parse it if it needs to
    config_path = os.path.join(gen_folder, "config.yaml")
    with open(config_path, "r") as f:
        config_data = f.read()
    checklist.append("config.yaml")

    # algorithm.py: this is Python source code, not data, we just read it as plain text instead of trying to parse it
    algorithm_path = os.path.join(gen_folder, "algorithm.py")
    with open(algorithm_path, "r") as f:
        algorithm_data = f.read()
    checklist.append("algorithm.py")

    # training_logs.txt: returned as raw text
    training_logs_path = os.path.join(gen_folder, "training_logs.txt")
    with open(training_logs_path, "r") as f:
        training_logs_data = f.read()
    checklist.append("training_logs.txt")

    # tournament_table.csv: this lives in shared results/folder, not inside of the gen folder, because it covers ALL generations.
    # ';'-delimited, no header - matches exactly what run_tournament.py writes.
    tournament_path = os.path.join(RESULTS_DIR, "tournament_table.csv")
    if not os.path.isfile(tournament_path):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        open(tournament_path, "w").close()
    tournament_data = _read_tournament_table(tournament_path)
    checklist.append("tournament_table.csv")

    standings_path = os.path.join(RESULTS_DIR, "tournament_results.csv")
    if os.path.isfile(standings_path):
        with open(standings_path, "r") as f:
            tournament_results = f.read()
        checklist.append("tournament_results.csv")
    else:
        tournament_results = ""


    # model_score_history.csv: returned as raw text
    score_history_path = os.path.join(RESULTS_DIR, "model_score_history.csv")
    if not os.path.isfile(score_history_path):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        open(score_history_path, "w").close()
    with open(score_history_path, "r") as f:
        score_history_data = f.read()
    checklist.append("model_score_history.csv")

    logger.info(
        "[gen %s] aggregating data:\n%s",
        gen_id,
        "\n".join(f"  ✔ {name}" for name in checklist),
    )

    return {
        "config": config_data,
        "algorithm": algorithm_data,
        "training_logs": training_logs_data,
        "tournament_table": tournament_data,
        "tournament_results": tournament_results,
        "model_score_history": score_history_data,
    }


def _read_tournament_table(path: str) -> list:
    """Reads the ';'-delimited, headerless tournament table into a list of lists."""
    with open(path, "r") as f:
        rows = [line.strip().split(";") for line in f if line.strip()]
    return [[None if v == "None" else int(v) for v in row] for row in rows]


if __name__ == "__main__":
    # Simple manual test hook.
    result = run_aggregate_data(gen_id=0)
    print(result)
