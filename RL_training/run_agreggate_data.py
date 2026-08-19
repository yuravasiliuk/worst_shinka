import os
import csv
import yaml  # pip install pyyaml if not already installed

RESULTS_DIR = "results" # Base folder where all generation results live


def run_agreggate_data(gen_id: int) -> dict:
    """
    Reads and returns the following files for a given generation:
      - config.yaml              (parsed as a dict)
      - algorithm.py             (returned as raw source text)
      - tournament_table.csv     (parsed as a list of row-dicts)
      - model_score_history.csv  (parsed as a list of row-dicts)

    Args:
        gen_id: generation number (e.g. 1 -> results/gen1)

    Returns:
        dict with keys: "config", "algorithm", "tournament_table", "model_score_history"
    """

    #Path to this generation's folder
    gen_folder = os.path.join(RESULTS_DIR, f"gen{gen_id}")


    # config.yaml:  structured data, so we parse it into a dict
    config_path = os.path.join(gen_folder, "config.yaml")
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    # algorithm.py: this is Python source code, not data, we just read it as plain text instead of trying to parse it 
    algorithm_path = os.path.join(gen_folder, "algorithm.py")
    with open(algorithm_path, "r") as f:
        algorithm_data = f.read()

    # tournament_table.csv: this lives in shared results/folder, not inside of the gen folder, because it covers ALL generations.
    # might become YAML instead of CSV - I will keep it as CSV for now, but if we switch to YAML, this will need to be updated.
    tournament_path = os.path.join(RESULTS_DIR, "tournament_table.csv")
    tournament_data = _read_csv(tournament_path)

    # model_score_history.csv
    score_history_path = os.path.join(RESULTS_DIR, "model_score_history.csv")
    score_history_data = _read_csv(score_history_path)

    return {
        "config": config_data,
        "algorithm": algorithm_data,
        "tournament_table": tournament_data,
        "model_score_history": score_history_data,
    }


def _read_csv(path: str) -> list:
    """Reads a CSV file into a list of dicts (one dict per row, keyed by header)."""
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


if __name__ == "__main__":
    # Simple manual test hook.
    result = run_agreggate_data(gen_id=0)
    print(result)