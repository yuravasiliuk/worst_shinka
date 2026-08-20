import os
import shutil

from train import train # function name might be different
from utils import RESULTS_DIR

def run_training(gen_id: int, config: str, algorithm: str) -> None:

    gen_folder = os.path.join(RESULTS_DIR, f"gen_{gen_id}")
    os.makedirs(gen_folder, exist_ok=True)
    # Copy the config file to the generation folder

    config_test = os.path.join(gen_folder, 'config.yaml')
    algorithm_test = os.path.join(gen_folder, 'algorithm.py')

    shutil.copy(config, config_test)
    shutil.copy(algorithm, algorithm_test)

    model_output_path = os.path.join(gen_folder, 'model.pt')

    train(
        gen_id=gen_id,
        config_path=config_test,
        algorithm_path=algorithm_test,
        model_output_path=model_output_path
    )

if __name__ == "__main__":
    # Manual test hook: retrains gen_0 using its own current config/algorithm.
    # Copied to a neutral path first since run_training(gen_id=0, ...) would
    # otherwise copy gen_0's files onto themselves (shutil.SameFileError).
    _sample_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_sample_config.yaml")
    _sample_algorithm = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_sample_algorithm.py")
    shutil.copy(os.path.join(RESULTS_DIR, "gen_0", "config.yaml"), _sample_config)
    shutil.copy(os.path.join(RESULTS_DIR, "gen_0", "algorithm.py"), _sample_algorithm)
    try:
        run_training(gen_id=0, config=_sample_config, algorithm=_sample_algorithm)
    finally:
        os.remove(_sample_config)
        os.remove(_sample_algorithm)