import os

from utils import RESULTS_DIR


def run_agreggate_data(gen_id: int) -> dict:
    """
    Reads and returns the following files for a given generation:
      - config.yaml              (raw text)
      - algorithm.py             (raw text)
      - training_logs.txt        (raw text)
      - tournament_table.csv     (parsed as a list of lists, None or int)
      - model_score_history.csv  (raw text)

    Args:
        gen_id: generation number (e.g. 1 -> results/gen_1)

    Returns:
        dict with keys: "config", "algorithm", "training_logs", "tournament_table", "model_score_history"
    """

    #Path to this generation's folder
    gen_folder = os.path.join(RESULTS_DIR, f"gen_{gen_id}")


    # config.yaml: returned as raw text, the caller can parse it if it needs to
    config_path = os.path.join(gen_folder, "config.yaml")
    with open(config_path, "r") as f:
        config_data = f.read()

    # algorithm.py: this is Python source code, not data, we just read it as plain text instead of trying to parse it
    algorithm_path = os.path.join(gen_folder, "algorithm.py")
    with open(algorithm_path, "r") as f:
        algorithm_data = f.read()

    # training_logs.txt: returned as raw text
    training_logs_path = os.path.join(gen_folder, "training_logs.txt")
    with open(training_logs_path, "r") as f:
        training_logs_data = f.read()

    # tournament_table.csv: this lives in shared results/folder, not inside of the gen folder, because it covers ALL generations.
    # ';'-delimited, no header - matches exactly what run_tournament.py writes.
    tournament_path = os.path.join(RESULTS_DIR, "tournament_table.csv")
    tournament_data = _read_tournament_table(tournament_path)

    # model_score_history.csv: returned as raw text
    score_history_path = os.path.join(RESULTS_DIR, "model_score_history.csv")
    with open(score_history_path, "r") as f:
        score_history_data = f.read()

    return {
        "config": config_data,
        "algorithm": algorithm_data,
        "training_logs": training_logs_data,
        "tournament_table": tournament_data,
        "model_score_history": score_history_data,
    }


def _read_tournament_table(path: str) -> list:
    """Reads the ';'-delimited, headerless tournament table into a list of lists."""
    with open(path, "r") as f:
        rows = [line.strip().split(";") for line in f if line.strip()]
    return [[None if v == "None" else int(v) for v in row] for row in rows]


if __name__ == "__main__":
    # Simple manual test hook.
    result = run_agreggate_data(gen_id=0)
    print(result)
