from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from inference_set_design.tasks.base_task import BaseTask
from inference_set_design.utils.metrics import get_ensemble_std
from inference_set_design.utils.misc import SimpleDataloader


class Mol3DCompoundAcquisition(BaseTask):
    @property
    def input_size(self):
        return self.explr_xs.shape[1]

    @property
    def output_size(self):
        return 1  # Only scalar regression is supported

    @property
    def n_classes(self):
        return None

    def load_data(self):

        explr_df = pd.read_parquet(Path(self.cfg.data_path) / "explorable.parquet")
        valid_df = pd.read_parquet(Path(self.cfg.data_path) / "valid.parquet")
        test_df = pd.read_parquet(Path(self.cfg.data_path) / "test.parquet")

        # Reduce the number of compounds to use
        if self.cfg.n_explorable_cmpds is not None:
            assert self.cfg.n_explorable_cmpds <= explr_df.shape[0]
        self.n_explr_cmpds = explr_df.shape[0] if self.cfg.n_explorable_cmpds is None else self.cfg.n_explorable_cmpds
        explr_df = explr_df.iloc[: self.n_explr_cmpds]

        # Extract xs and ys from dataframes and send to device (small dataset, easily fits in memory)
        self.explr_xs = torch.tensor(np.vstack(explr_df[self.cfg.emb_name].to_numpy()), dtype=torch.float).to(
            self.device
        )
        self.valid_xs = torch.tensor(np.vstack(valid_df[self.cfg.emb_name].to_numpy()), dtype=torch.float).to(
            self.device
        )
        self.test_xs = torch.tensor(np.vstack(test_df[self.cfg.emb_name].to_numpy()), dtype=torch.float).to(self.device)

        self.explr_ys = torch.tensor(np.array(explr_df[self.cfg.label_name].values), dtype=torch.float).to(self.device)
        self.valid_ys = torch.tensor(np.array(valid_df[self.cfg.label_name].values), dtype=torch.float).to(self.device)
        self.test_ys = torch.tensor(np.array(test_df[self.cfg.label_name].values), dtype=torch.float).to(self.device)

        # Reveal compounds that are initially part of the training set (could be 0)
        if self.reveal_all:
            self.revealed_cmpds = np.ones((self.n_explr_cmpds,))
        else:
            self.revealed_cmpds = np.zeros((self.n_explr_cmpds,))
            self.revealed_cmpds[: self.cfg.n_init_train_cmpds] = 1

        # Update acquisition mask
        self.acquisition_mask = self.get_acquisition_mask()

    def get_acquisition_mask(self):
        return 1.0 - self.revealed_cmpds

    def label_batch(self, acquisition_idxs: List[int]):
        for i in acquisition_idxs:
            self.revealed_cmpds[i] = 1
        self.acquisition_mask = self.get_acquisition_mask()

    def build_dataloader(self, xs: torch.Tensor, ys: torch.Tensor, shuffle: bool = False):
        return SimpleDataloader(
            xs=xs,
            ys=ys,
            batch_size=self.model_cfg.train_batch_size,
            shuffle=shuffle,
        )

    def get_explorable_dataloader(self) -> DataLoader:
        return self.build_dataloader(xs=self.explr_xs, ys=self.explr_ys, shuffle=False)

    def get_hidden_dataloader(self, acquisition_mask: np.ndarray) -> DataLoader:
        available_acq_idxs = np.argwhere(acquisition_mask == 1).flatten()

        if len(available_acq_idxs) > 0:
            hidden_loader = self.build_dataloader(
                xs=self.explr_xs[available_acq_idxs], ys=self.explr_ys[available_acq_idxs], shuffle=False
            )
        else:
            hidden_loader = None

        return hidden_loader

    def get_train_dataloader(self, acquisition_mask: np.ndarray) -> DataLoader:
        revealed_acq_idxs = np.argwhere(acquisition_mask == 0).flatten()

        if len(revealed_acq_idxs) > 0:
            train_loader = self.build_dataloader(
                xs=self.explr_xs[revealed_acq_idxs], ys=self.explr_ys[revealed_acq_idxs], shuffle=True
            )
        else:
            train_loader = None

        return train_loader

    def get_valid_dataloader(self) -> DataLoader:
        return self.build_dataloader(xs=self.valid_xs, ys=self.valid_ys, shuffle=False)

    def get_test_dataloader(self) -> DataLoader:
        return self.build_dataloader(xs=self.test_xs, ys=self.test_ys, shuffle=False)

    def get_batch_dataloader(self, batch: np.ndarray):
        if len(batch) > 0:
            batch_loader = self.build_dataloader(xs=self.explr_xs[batch], ys=self.explr_ys[batch], shuffle=False)
        else:
            batch_loader = None

        return batch_loader

    def get_acquisition_scores(self, ensemble_preds: np.ndarray, x_idxs: np.ndarray):
        n_samples, n_models = ensemble_preds.shape

        # TODO: MAKE SURE THE ORDERING FROM THE DATALOADER THAT PRODUCES class_probs
        # IS THE SAME AS THE ORDERING OF COMPOUNDS IN THE ACQUISITION MASKS

        # Get stardand deviation of ensemble predictions for compounds
        std = get_ensemble_std(ensemble_preds)

        # Prepare the acquisition scores
        acquisition_scores = {
            "std": std,
        }
        acquisition_metrics = {
            "max_std": float(np.max(std)),
            "mean_std": float(np.mean(std)),
        }

        return acquisition_metrics, acquisition_scores

    def compute_metrics(self, preds: torch.Tensor, labels: torch.Tensor, data_cut: str):
        mse = torch.mean((preds - labels) ** 2)
        mae = torch.mean(torch.abs(preds - labels))
        return {f"mse_{data_cut}": mse.detach().cpu().item(), f"mae_{data_cut}": mae.detach().cpu().item()}
