import numpy as np
import statistics as stats

#performances are taken from each generation folder
#also somehow we should keep track of number of children of each model 
#maybe to create a file with structure: (model_id, generation_number, number_children, performance)

def calculate_s(performances, lmbd):
    alpha_0 = stats.median(performances)
    s = 1 / (1 + np.exp(-1*lmbd*(performances - alpha_0)))
    return s

def calculate_h(N):
    h = 1 / (1 + N)
    return h

def calculate_w(s, h):
    w = s*h
    return w

def calculate_p(w):
    p = w / np.sum(w)
    return p

def select_parent_ids(k, ids, performances, N, lmbd):
    s = calculate_s(performances, lmbd)
    h = calculate_h(N)
    w = calculate_w(s, h)
    p = calculate_p(w)

    selected_parent_ids = np.random.choice(ids, size = k, replace = False, p = p)
    return selected_parent_ids
