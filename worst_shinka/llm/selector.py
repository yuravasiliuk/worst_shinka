from .pool_reader import Reader_Model_Pool
import numpy as np
import yaml

models_probabilities = {}

class Selector_LLM():
    def __init__(self, model_ids, lower_threshold=0.1, upper_threshold=0.9, delta_probability=0.05):
        """
        Initialization of probability distribution of models. 
        At the beginning the distribution is uniform.
        """
        self.model_ids = model_ids 
        self.probabilities = [1/len(model_ids) for _ in range(len(model_ids))]
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        self.delta_probability = delta_probability

    def select_models(self):
        """
        Method selects two models based on corresponding probability distribution.
        Method return two model ids.
        """
        two_selected_model_ids = np.random.choice(self.model_ids, size = 2, replace = False, p = self.probabilities)
        return two_selected_model_ids

    def update_probabilities(self, model_id_1, model_id_2, who_won): 
        """
        Method updates probabilities of two arguing models. 
        The winner gets a higher probability of being selected for the next discussion, symmetrically the loser obtains a lower probability.
        The change in probabilities is constant == self.delta_probability.
        Change in probabilities is possible if and only if new probabilities are within the preset lower and upper bound.
        It is assumed that the model with id model_id_1 has won if who_won == 1, otherwise it is assumed that the second model has won.
        """
        index_1 = self.model_ids.index(model_id_1)
        index_2 = self.model_ids.index(model_id_2)
        if who_won == 1:
            new_prob_1 = self.probabilities[index_1] + self.delta_probability
            new_prob_2 = self.probabilities[index_2] - self.delta_probability
            if (new_prob_1 < self.upper_threshold) and (new_prob_2 > self.lower_threshold):
                self.probabilities[index_1] = new_prob_1
                self.probabilities[index_2] = new_prob_2
        else:
            new_prob_1 = self.probabilities[index_1] - self.delta_probability
            new_prob_2 = self.probabilities[index_2] + self.delta_probability
            if (new_prob_1 > self.lower_threshold) and (new_prob_2 < self.upper_threshold):
                self.probabilities[index_1] = new_prob_1
                self.probabilities[index_2] = new_prob_2

# reader = Reader_Model_Pool()
# reader.read_model_ids_from_yaml("Test")
# model_ids = reader.get_model_ids()
# print("fetched valid model ids: ", model_ids)
# selector = Selector_LLM(model_ids, 0.2, 0.8, 0.05)
# selected = selector.select_models()
# print("selected two models for the first iteration: ", selected)