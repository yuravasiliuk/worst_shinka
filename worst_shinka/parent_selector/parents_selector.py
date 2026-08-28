import numpy as np
import statistics as stats

#performances are taken from each generation folder
#also somehow we should keep track of number of children of each model 
#maybe to create a file with structure: (model_id, generation_number, number_children, performance)

class Selector_Parents():
    def __init__(self, lmbd=1.0):
        self.lmbd = lmbd 
        self.N = {} #dictionary that keeps track of number of children for each model_id

    def update_N(self, selected_parent_ids):
        for id in selected_parent_ids:
            if id in self.N:
                self.N[id] += 1
            else:
                self.N[id] = 1
        
    def calculate_s(self, performances):
        alpha_0 = stats.median(performances)
        s = 1 / (1 + np.exp(-1*self.lmbd*(np.array(performances) - alpha_0)))
        return s

    def calculate_h(self, ids):
        h = []
        for id in ids:
            if id in self.N:
                h.append(1 / (1 + self.N[id]))
            else:
                h.append(1)
        return h

    def calculate_w(self, s, h):
        w = s*h
        return w

    def calculate_p(self, w):
        p = w / np.sum(w)
        return p

    def select_parent_ids(self, k, ids, performances):
        """
        k - number of parents to select
        ids - ids of possible parents to choose from (must be stable identifiers, e.g. "gen_3" -
              not positions into a list that can reorder/shrink across calls, since self.N persists
              across calls and is keyed by these ids)
        performances - scores of parents, corresponding to ids
        """
        s = self.calculate_s(performances)
        h = self.calculate_h(ids)
        w = self.calculate_w(s, h)
        p = self.calculate_p(w)

        selected_parent_ids = np.random.choice(ids, size = k, replace = False, p = p).tolist()

        return selected_parent_ids
