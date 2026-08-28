from pathlib import Path
import csv
import shutil


RESULTS_DIR = Path("results")
GEN0_DIR = RESULTS_DIR / "gen0"


def reset_results():
    RESULTS_DIR.mkdir(exist_ok=True)
    GEN0_DIR.mkdir(exist_ok=True)

    # Delete every generation folder/file except gen0
    for item in RESULTS_DIR.iterdir():
        if item.name == "gen0":
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    # Reset shared result tables
    with open(RESULTS_DIR / "tournament_table.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "model", "score", "elo"])

    with open(RESULTS_DIR / "model_score_history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "model", "score", "elo"])


if __name__ == "__main__":
    reset_results()
    print("Results reset successfully. gen0 preserved.")


