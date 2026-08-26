from fileinput import filename
import json
import yaml

class Reader_Model_Pool():
    def __init__(self):
        self.model_ids = []

    def read_model_ids_from_yaml(self, NAME_OF_EXPERIMENT):
        with open(f"./results/{NAME_OF_EXPERIMENT}/models.json", "r") as file:
            data = json.load(file)

        self.model_ids = [model["id"] for model in data["models"] if model.get("status") == "valid"]

    def get_model_ids(self):
        return self.model_ids