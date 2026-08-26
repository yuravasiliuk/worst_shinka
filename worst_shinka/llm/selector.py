from .pool_reader import Reader_Model_Pool
import numpy as np
import yaml

class Selector_LLM():
    def __init__(self, model_ids, low_threshold=0.1, upper_threshold=0.9, delta_probability=0.05):
        self.model_ids = model_ids 
        self.probabilities = [1/len(model_ids) for _ in range(len(model_ids))]
        self.low_threshold = low_threshold
        self.upper_threshold = upper_threshold
        self.delta_probability = delta_probability

    def select_models(self):
        two_selected_model_ids = np.random.choice(self.model_ids, size = 2, replace = False, p = self.probabilities)
        return two_selected_model_ids

    def check_prob_in_bounds(self, model_id_index):
        return self.probabilities[model_id_index] < self.upper_threshold and self.probabilities[model_id_index] > self.low_threshold

    def update_probabilities(self, model_id_1, model_id_2, who_won): 
        index_1 = self.model_ids.index(model_id_1)
        index_2 = self.model_ids.index(model_id_2)
        if self.check_prob_in_bounds(index_1) and self.check_prob_in_bounds(index_2):
            if who_won == 1:
                self.probabilities[index_1] += self.delta_probability
                self.probabilities[index_2] -= self.delta_probability
            else:
                self.probabilities[index_2] += self.delta_probability
                self.probabilities[index_1] -= self.delta_probability

reader = Reader_Model_Pool()
reader.read_model_ids_from_yaml("Test")
model_ids = reader.get_model_ids()
print("fetched valid model ids: ", model_ids)
selector = Selector_LLM(model_ids, 0.2, 0.8, 0.05)
selected = selector.select_models()
print("selected two models for the first iteration: ", selected)