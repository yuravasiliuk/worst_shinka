import os 
import shutil 

from train import train # function name might be different 

RESULTS_DIR = "results"

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
        config_path=config_test,
        algorithm_path=algorithm_test,
        model_output_path=model_output_path
    )

if __name__ == "__main__":
    run_training(gen_id=0, config='sample_config_yaml', algorithm='sample_algorithm.py')