from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import (
    cosine_distances,
    euclidean_distances,
    pairwise_distances,
)

from inference_set_design.agents.base_agent import BaseAgent
from inference_set_design.agents.config import ActiveAgentConfig
from inference_set_design.tasks.base_task import BaseTask


class DiversityBaseAgent(BaseAgent):

    def __init__(
        self,
        emb_name: str,
        data_path: str,
        n_explorable_cmpds: int,
        agent_cfg: ActiveAgentConfig,
        task: BaseTask,
        acquisition_batch_size: int,
        device: str,
        log_path: Path,
    ):
        super().__init__(
            agent_cfg=agent_cfg,
            task=task,
            acquisition_batch_size=acquisition_batch_size,
            device=device,
            log_path=log_path,
        )

        # Load explorable data
        explr_df = pd.read_parquet(Path(data_path) / "explorable.parquet")

        # Reduce the number of compounds to use
        if n_explorable_cmpds is not None:
            assert n_explorable_cmpds <= explr_df.shape[0]
        self.n_explr_cmpds = explr_df.shape[0] if n_explorable_cmpds is None else n_explorable_cmpds
        explr_df = explr_df.iloc[: self.n_explr_cmpds]

        # Extract xs from dataframes
        explr_xs = np.vstack(explr_df[emb_name].to_numpy())
        self.distances = self.compute_distances(explr_xs)

    def compute_distances(self):
        raise NotImplementedError("computed_distances() needs to be implemented by the child class")

    def acquisition_function(self, available_indices: np.ndarray, acquisition_scores: dict):
        # Find acquired indices
        acquired_indices = np.ones(len(self.distances))
        acquired_indices[available_indices] = 0
        acquired_indices = np.where(acquired_indices)[0]

        # Compute average distance of explorable samples to acquired samples
        acquisition_scores = self.distances[acquired_indices].mean(axis=0)

        # Sort samples based on acquisition scores and select a batch
        idx_score = [(idx, acquisition_scores[idx]) for idx in available_indices]
        idx_score = sorted(idx_score, key=lambda x: x[1], reverse=True)
        acquisition_batch = np.array([idx for idx, _ in idx_score[: self.acquisition_batch_size]])
        return acquisition_batch


class DiveristyTanimotoAgent(DiversityBaseAgent):
    def compute_distances(self, embeddings: np.ndarray):
        return pairwise_distances(embeddings, embeddings, metric="jaccard", n_jobs=-1)


class DiveristyCosineAgent(DiversityBaseAgent):
    def compute_distances(self, embeddings: np.ndarray):
        return cosine_distances(embeddings, embeddings)


class DiveristyEuclideanAgent(DiversityBaseAgent):
    def compute_distances(self, embeddings: np.ndarray):
        return euclidean_distances(embeddings, embeddings)
