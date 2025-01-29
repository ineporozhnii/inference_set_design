import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from inference_set_design.agents.base_agent import BaseAgent


class BALDAgent(BaseAgent):
    def get_explorable_predictions(self, model: nn.Module, dataloader: DataLoader):

        all_class_probs_mc = []
        for i in range(self.cfg.mc_iterations):
            all_class_probs, all_labels, all_x_idxs = self.task.predict(
                model=model, dataloader=dataloader, mc_dropout=True
            )
            all_class_probs_mc.append(all_class_probs)

        all_class_probs_mc = torch.stack(all_class_probs_mc)
        if self.task.cfg.emb_name == "molgps_fps":
            n_cmpds = all_class_probs_mc.shape[1]
            n_genes = all_class_probs_mc.shape[2]
            all_class_probs_mc = all_class_probs_mc.reshape(
                self.cfg.mc_iterations, n_cmpds, n_genes, self.task.n_classes
            )
        all_class_probs_mean = all_class_probs_mc.mean(dim=0)
        H = (-all_class_probs_mean * torch.log(all_class_probs_mean + 1e-9)).sum(dim=-1)
        E = -torch.mean(torch.sum(all_class_probs_mc * torch.log(all_class_probs_mc + 1e-9), dim=-1), dim=0)
        self.bald_scores = (H - E).detach().cpu().numpy()

        if self.task.cfg.emb_name == "molgps_fps":
            self.bald_scores = self.bald_scores.reshape(n_cmpds, n_genes)
            self.bald_scores = np.mean(self.bald_scores, axis=1)

        return all_class_probs_mean, all_labels, all_x_idxs

    def acquisition_function(self, available_indices: np.ndarray, acquisition_scores: dict):
        # Use BALD scores
        acquisition_scores = self.bald_scores
        # Sort samples based on acquisition scores and select a batch
        idx_score = [(idx, acquisition_scores[idx]) for idx in available_indices]
        idx_score = sorted(idx_score, key=lambda x: x[1], reverse=True)
        acquisition_batch = np.array([idx for idx, _ in idx_score[: self.acquisition_batch_size]])
        return acquisition_batch
