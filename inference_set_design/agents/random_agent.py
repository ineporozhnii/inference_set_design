import numpy as np

from inference_set_design.agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    def acquisition_function(self, available_indices: np.ndarray, acquisition_scores: dict):
        return np.random.choice(available_indices, size=self.acquisition_batch_size, replace=False)
