import numpy as np

from inference_set_design.agents.base_agent import BaseAgent


class GridAgent(BaseAgent):
    def acquisition_function(self, available_indices: np.ndarray, acquisition_scores: dict):
        return available_indices[: self.acquisition_batch_size]
