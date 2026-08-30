from __future__ import annotations

import numpy as np




class Selector_Parents:
    """Weighted sampling of distinct parents based on tournament strength."""


    def __init__(self, temperature: float = 0.35, exploration: float = 0.2, rng=None):
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        if not 0 <= exploration < 1:
            raise ValueError("exploration must be in [0, 1)")
        self.temperature = temperature
        self.exploration = exploration
        self.rng = rng if rng is not None else np.random.default_rng()
        self.N = {}


    def update_N(self, selected_parent_ids):
        for parent_id in selected_parent_ids:
            self.N[parent_id] = self.N.get(parent_id, 0) + 1


    def calculate_p(self, performances):
        values = np.asarray(performances, dtype=float)
        if values.ndim != 1 or len(values) == 0:
            raise ValueError("performances must be a non-empty one-dimensional sequence")
        finite = np.isfinite(values)
        if not finite.any():
            return np.full(len(values), 1.0 / len(values))
        floor = float(np.min(values[finite]))
        values = np.where(finite, values, floor)
        spread = float(np.std(values))
        if spread < 1e-12:
            probabilities = np.full(len(values), 1.0 / len(values))
        else:
            logits = (values - float(np.max(values))) / (spread * self.temperature)
            weights = np.exp(np.clip(logits, -50, 0))
            probabilities = weights / weights.sum()
        # A small exploration share prevents permanently excluding weaker models.
        uniform = np.full(len(values), 1.0 / len(values))
        return (1.0 - self.exploration) * probabilities + self.exploration * uniform


    def select_parent_ids(self, k, ids, performances):
        if len(ids) != len(performances):
            raise ValueError("ids and performances must have the same length")
        if not ids:
            raise ValueError("at least one parent candidate is required")
        count = min(max(1, int(k)), len(ids))
        probabilities = self.calculate_p(performances)
        return self.rng.choice(ids, size=count, replace=False, p=probabilities).tolist()
