from functools import cached_property
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torcheval.metrics import BinaryAUROC

from inference_set_design.models.mlps import MultiTaskMLP
from inference_set_design.tasks.base_task import BaseTask
from inference_set_design.utils.metrics import get_classification_uncertainty
from inference_set_design.utils.misc import SimpleDataloader


class RxRx3MapAcquisition(BaseTask):
    @property
    def input_size(self):
        return self.explr_xs.shape[1]

    @cached_property
    def n_classes(self):
        n_classes = len(np.unique(self.explr_ys.cpu().numpy()))
        assert (
            n_classes == 2
        ), f"MolEmbedMapAcquisition is supposed to be binary classification, but n_classes={n_classes}."
        return n_classes

    def load_data(self):

        explr_df = pd.read_parquet(Path(self.cfg.data_path) / "explorable.parquet")
        valid_df = pd.read_parquet(Path(self.cfg.data_path) / "valid.parquet")
        test_df = pd.read_parquet(Path(self.cfg.data_path) / "test.parquet")

        # Extracting gene information from dataframe
        gene_cols = explr_df.columns[5:]
        assert "folds" not in gene_cols
        assert "smiles" not in gene_cols
        assert "molgps_fps" not in gene_cols
        assert "master_idx" not in gene_cols
        assert "rec_id" not in gene_cols
        self.num_genes = len(gene_cols)

        # Reduce the number of compounds to use
        if self.cfg.n_explorable_cmpds is not None:
            assert self.cfg.n_explorable_cmpds <= explr_df.shape[0]
        self.n_explr_cmpds = explr_df.shape[0] if self.cfg.n_explorable_cmpds is None else self.cfg.n_explorable_cmpds
        explr_df = explr_df.iloc[: self.n_explr_cmpds]

        # Extract xs and ys from dataframes and send to device
        self.explr_xs = torch.tensor(np.vstack(explr_df[self.cfg.emb_name].to_numpy()), dtype=torch.float).to(
            self.device
        )
        self.valid_xs = torch.tensor(np.vstack(valid_df[self.cfg.emb_name].to_numpy()), dtype=torch.float).to(
            self.device
        )
        self.test_xs = torch.tensor(np.vstack(test_df[self.cfg.emb_name].to_numpy()), dtype=torch.float).to(self.device)

        self.explr_ys = torch.tensor(np.array(explr_df.loc[:, gene_cols].values), dtype=torch.long).to(self.device)
        self.valid_ys = torch.tensor(np.array(valid_df.loc[:, gene_cols].values), dtype=torch.long).to(self.device)
        self.test_ys = torch.tensor(np.array(test_df.loc[:, gene_cols].values), dtype=torch.long).to(self.device)

        # All genes are revealed
        self.revealed_genes = np.ones((self.num_genes,))

        if self.reveal_all:
            self.revealed_cmpds = np.ones((self.n_explr_cmpds,))
        else:
            self.revealed_cmpds = np.zeros((self.n_explr_cmpds,))
            self.revealed_cmpds[: self.cfg.n_init_train_cmpds] = 1

        # Update acquisition mask
        self.acquisition_mask = self.get_acquisition_mask()

    def initialise_model(self):
        if self.model_cfg.model_name == "MultiTaskMLP":
            kwargs = {
                "input_size": self.input_size,
                "trunk_hidden_size": self.model_cfg.trunk_hidden_size,
                "n_trunk_res_block": self.model_cfg.n_trunk_res_block,
                "task_hidden_size": self.model_cfg.task_hidden_size,
                "n_task_layers": self.model_cfg.n_task_layers,
                "n_tasks": self.num_genes,
                "output_size": self.n_classes,
                "skip_connections": self.model_cfg.skip_connections,
                "dropout": self.model_cfg.dropout,
            }
            assert self.model_cfg.num_ensmbl_members is None, "MultiTaskMLP Ensemble is not supported yet."
            model = MultiTaskMLP(**kwargs).to(self.device)
        else:
            raise NotImplementedError(
                f"Model {self.model_cfg.model_name} is not supported for MolEmbedMapAcquisition task."
            )
        return model

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
        explorable_loader = self.build_dataloader(xs=self.explr_xs, ys=self.explr_ys, shuffle=False)
        return explorable_loader

    def get_hidden_dataloader(self, acquisition_mask: np.ndarray) -> DataLoader:
        available_acq_idxs = np.argwhere(acquisition_mask == 1).flatten()

        if len(available_acq_idxs) > 0:
            explorable_hidden_loader = self.build_dataloader(
                xs=self.explr_xs[available_acq_idxs], ys=self.explr_ys[available_acq_idxs], shuffle=False
            )
        else:
            explorable_hidden_loader = None

        return explorable_hidden_loader

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
        n_samples, n_genes, n_classes = class_probs.shape
        assert n_classes == self.n_classes

        class_probs = class_probs.reshape((n_samples * n_genes, self.n_classes))

        # Get the mean uncertainty for compounds
        uncertainty = get_classification_uncertainty(class_probs, n_classes=n_classes)

        uncertainty = uncertainty.reshape((n_samples, n_genes))
        uncertainty = np.mean(uncertainty, axis=1)

        # Prepare the acquisition scores
        acquisition_scores = {
            "uncertainty": uncertainty,
        }
        acquisition_metrics = {
            "max_uncertainty": float(np.mean(uncertainty)),
            "mean_uncertainty": float(np.max(uncertainty)),
        }

        return acquisition_metrics, acquisition_scores

    def compute_metrics(self, class_probs: torch.Tensor, labels: torch.Tensor, data_cut: str):
        class_preds = torch.argmax(class_probs, dim=-1)

        class_wise_acc = {}
        for y_i in range(self.n_classes):
            idxs = labels == y_i
            class_wise_acc[y_i] = torch.mean((class_preds[idxs] == labels[idxs]).float())

        acc = torch.mean((class_preds == labels).float())
        acc_balanced = torch.nanmean(torch.tensor(list(class_wise_acc.values())))

        # remove the nan values in the class_wise_acc for classes that are not present in the batch
        class_wise_acc = {k: v if not torch.isnan(v) else -1 for k, v in class_wise_acc.items()}
        acc_by_class = {f"acc_y{c}_{data_cut}": acc_c.detach().cpu().item() for c, acc_c in class_wise_acc.items()}

        # auroc
        global_auroc = BinaryAUROC(device=self.device, num_tasks=1)
        global_auroc.update(class_probs[:, :, 0].flatten(), labels.flatten())

        per_gene_auroc = BinaryAUROC(device=self.device, num_tasks=self.num_genes)
        per_gene_auroc.update(class_probs[:, :, 0].T, labels.T)
        per_gene_aurocs = per_gene_auroc.compute()
        median_gene_auroc = torch.median(per_gene_aurocs)
        max_gene_auroc = torch.max(per_gene_aurocs)
        min_gene_auroc = torch.min(per_gene_aurocs)

        return {
            f"acc_{data_cut}": acc.detach().cpu().item(),
            f"acc_balanced_{data_cut}": acc_balanced.detach().cpu().item(),
            **acc_by_class,
            f"auroc_{data_cut}": global_auroc.compute().detach().cpu().item(),
            f"median_gene_auroc_{data_cut}": median_gene_auroc.detach().cpu().item(),
            f"max_gene_auroc_{data_cut}": max_gene_auroc.detach().cpu().item(),
            f"min_gene_auroc_{data_cut}": min_gene_auroc.detach().cpu().item(),
        }

    def compute_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_hat = y_hat.flatten(end_dim=-2)
        y = y.flatten()

        if self.cfg.use_class_balancing_weights:
            class_counts = torch.bincount(y, minlength=self.n_classes)
            class_weights = class_counts.sum() / (class_counts.float() + 1e-6)
            return nn.functional.cross_entropy(input=y_hat, target=y, weight=class_weights).mean(dim=0)
        else:
            return nn.functional.cross_entropy(input=y_hat, target=y).mean(dim=0)
