from functools import cached_property
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from inference_set_design.tasks.base_task import BaseTask
from inference_set_design.utils.metrics import get_classification_uncertainty
from inference_set_design.utils.misc import SimpleDataloader


class QM9CompoundAcquisition(BaseTask):
    @property
    def input_size(self):
        return self.explr_xs.shape[1]

    @cached_property
    def n_classes(self):
        n_classes = len(np.unique(self.explr_ys.cpu().numpy()))
        assert n_classes == 2, f"QM9 is supposed to be binary classification, but n_classes={n_classes}."
        return n_classes

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

        self.explr_ys = torch.tensor(np.array(explr_df[self.cfg.label_name].values), dtype=torch.long).to(self.device)
        self.valid_ys = torch.tensor(np.array(valid_df[self.cfg.label_name].values), dtype=torch.long).to(self.device)
        self.test_ys = torch.tensor(np.array(test_df[self.cfg.label_name].values), dtype=torch.long).to(self.device)

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

    def get_acquisition_scores(self, class_probs: np.ndarray, x_idxs: np.ndarray):
        n_samples, n_classes = class_probs.shape
        assert n_classes == self.n_classes
        # TODO: MAKE SURE THE ORDERING FROM THE DATALOADER THAT PRODUCES class_probs
        # IS THE SAME AS THE ORDERING OF COMPOUNDS IN THE ACQUISITION MASKS

        # Get the mean uncertainty for compounds
        uncertainty = get_classification_uncertainty(class_probs, n_classes=n_classes)

        # Prepare the acquisition scores
        acquisition_scores = {
            "uncertainty": uncertainty,
        }
        acquisition_metrics = {
            "max_uncertainty": float(np.mean(uncertainty)),
            "mean_uncertainty": float(np.max(uncertainty)),
        }

        return acquisition_metrics, acquisition_scores
