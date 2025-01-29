from typing import Dict

import numpy as np

from inference_set_design.agents.base_agent import BaseAgent


class ActiveAgent(BaseAgent):
    def acquisition_function(self, available_indices: np.ndarray, acq_scores: Dict[str, np.ndarray]) -> np.ndarray:
        assert np.sum(self.cfg.acq_weights) == 1.0, "acq_weights must sum to 1.0"
        assert not (any([weight < 0.0 for weight in self.cfg.acq_weights])), "acq_weights must be non-negative"
        acq_comp = tuple(zip(self.cfg.acq_criteria, self.cfg.acq_weights))

        if self.cfg.acquisition_strategy == "greedy":
            # the picking order could potentially affect the result
            # but this is more likely the case when the proportions are unbalanced (which might be desirable)
            # to counteract this, we sort the selections by the number of indices to pick, and the method
            # with the fewest indices to select picks first
            acq_comp = sorted(acq_comp, key=lambda x: x[1])
            selected_indices = []
            for acq_criterion, weight in acq_comp:
                assert acq_criterion in list(acq_scores.keys()) + ["random"], (
                    f"acq_criterion {acq_criterion} not in acq_scores or 'random'",
                )
                # compose a batch of indices among available_indices
                # with the highest acquisition_scores
                n = int(self.acquisition_batch_size * weight)
                if n == 0:
                    continue

                indices = []
                if acq_criterion == "random" and n > 0:
                    indices = np.random.choice(available_indices, size=n, replace=False)
                else:
                    acquisition_scores = acq_scores[acq_criterion]
                    idx_score = [(idx, acquisition_scores[idx]) for idx in available_indices]
                    idx_score = sorted(idx_score, key=lambda x: x[1], reverse=True)
                    indices = np.array([idx for idx, _ in idx_score[:n]])

                assert set(indices).issubset(set(available_indices))
                selected_indices.append(indices)
                available_indices = np.setdiff1d(available_indices, indices)

            acquisition_batch = np.concatenate(selected_indices).astype(int)

        else:
            raise NotImplementedError()

        return acquisition_batch
