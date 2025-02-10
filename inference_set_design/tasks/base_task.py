import logging
import time
from abc import abstractmethod
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

from inference_set_design.models.config import ModelConfig
from inference_set_design.models.mlps import (
    MLP,
    MLPEnsemble,
    ResMLP,
    ResMLPEnsemble,
    count_parameters,
)
from inference_set_design.tasks.config import TaskConfig
from inference_set_design.utils.misc import EarlyStopper


class BaseTask:
    """
    The 'Task' class is responsible for implementing everything that goes into one active learning step.

    It is the prediction task that is being wrapped into the active learning loop.
    It is responsible for:
    - the dataloaders for all splits of interest:
        - train set: the 'acquired' set on which the modle is trained at each step
        - hidden set: the 'unseen' set that hasn't been acquired yet by the agent
        - explorable set: the union of the train and hidden sets (all examples that could be labeled)
        - test set: a separate set of examples that are used to evaluate the model but not for acquisition
        - valid set: any data split that is used for monitoring overfitting during training and trigger early stopping
        - batch set: the data obtained from the acquisition batch
    - a train() function, which takes as input a model, a train_loader and a val_loader, and returns a trained_model
    - a predict() function, which takes as input a model and a dataloader, and returns all the labels and predictions
    - a compute_metrics() function, which takes as input the class_probs and labels, and returns a dict of metrics
    - a get_early_stopper() function, which returns an object used to monitor overfitting

    The 'Task' is also responsible for handling the data and keeping track of which examples have been acquired.
    To that effect, it implements the following methods:
    - load_data(): loads the data from a given path
    - label_batch(): labels a batch of examples
    - get_acquisition_mask(): returns a boolean mask of the acquired examples
    - get_acquisition_scores(): returns the acquisition scores for a given set of predictions
    """

    # Methods that must be implemented by subclasses:
    # ------------------------------------------------

    @abstractmethod
    def load_data(self, data_path: str) -> None:
        raise NotImplementedError()

    @abstractmethod
    def label_batch(self, batch: List[int]) -> None:
        raise NotImplementedError()

    @abstractmethod
    def get_acquisition_mask(self) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def get_explorable_dataloader(self) -> DataLoader:
        raise NotImplementedError()

    @abstractmethod
    def get_hidden_dataloader(self) -> DataLoader:
        raise NotImplementedError()

    @abstractmethod
    def get_train_dataloader(self) -> DataLoader:
        raise NotImplementedError()

    @abstractmethod
    def get_valid_dataloader(self) -> DataLoader:
        raise NotImplementedError()

    @abstractmethod
    def get_test_dataloader(self) -> DataLoader:
        raise NotImplementedError()

    @abstractmethod
    def get_batch_dataloader(self) -> DataLoader:
        raise NotImplementedError()

    @abstractmethod
    def get_acquisition_scores(self, class_probs: np.ndarray, x_idxs: np.ndarray) -> dict:
        raise NotImplementedError()

    @property
    def n_classes(self):
        raise NotImplementedError()

    @property
    def input_size(self):
        raise NotImplementedError()

    @property
    def output_size(self):
        raise NotImplementedError()

    # Methods that can be optionally implemented by subclasses:
    # ---------------------------------------------------------

    def __init__(
        self,
        task_cfg: TaskConfig,
        model_cfg: ModelConfig,
        log_path: Path,
        logger: logging.Logger,
        device: torch.device,
        reveal_all: bool,
    ):
        self.cfg = task_cfg
        self.model_cfg = model_cfg
        self.log_path = log_path
        self.logger = logger
        self.device = device
        self.reveal_all = reveal_all
        self.use_ensmbl = False
        if self.model_cfg.num_ensmbl_members is not None:
            self.use_ensmbl = True

        self.load_data()
        tmp_model = self.initialise_model()
        logger.info(f"Model size: {count_parameters(tmp_model):,} params")
        logger.info(f"Model architecture: {tmp_model}")

    def initialise_model(self):
        """
        Here we instantiate a few generic model architectures.
        Tasks that require a specific model architecture should override this method.
        """
        if self.model_cfg.model_name == "ResMLP":
            kwargs = {
                "input_size": self.input_size,
                "hidden_size": self.model_cfg.hidden_size,
                "output_size": self.n_classes if self.n_classes is not None else self.output_size,
                "n_res_block": self.model_cfg.n_hidden_layers,
                "use_batch_norm": False,
                "dropout": self.model_cfg.dropout,
            }
            if self.model_cfg.num_ensmbl_members is not None:
                model = ResMLPEnsemble(
                    n_models=self.model_cfg.num_ensmbl_members,
                    **kwargs,
                ).to(self.device)
            else:
                model = ResMLP(**kwargs).to(self.device)

        elif self.model_cfg.model_name == "MLP":
            kwargs = {
                "input_size": self.input_size,
                "hidden_size": self.model_cfg.hidden_size,
                "output_size": self.n_classes if self.n_classes is not None else self.output_size,
                "n_hidden_layers": self.model_cfg.n_hidden_layers,
                "skip_connections": self.model_cfg.skip_connections,
            }
            if self.model_cfg.num_ensmbl_members is not None:
                model = MLPEnsemble(
                    n_models=self.model_cfg.num_ensmbl_members,
                    **kwargs,
                ).to(self.device)
            else:
                model = MLP(**kwargs).to(self.device)

        else:
            raise ValueError(
                f"Model {self.model_cfg.model_name} not supported for the BaseTask. "
                f"If your 'Task' uses a specialised model (e.g. Sphere), your Task class "
                f"should implement the 'initialise_model' method."
            )
        return model

    def save_model(self, path: Path, model: nn.Module):
        torch.save(model.state_dict(), path / "model.pt")

    def compute_metrics(self, class_probs: torch.Tensor, labels: torch.Tensor, data_cut: str):
        class_preds = torch.argmax(class_probs, dim=-1)
        acc = torch.mean((class_preds == labels).float(), dim=0)
        return {f"acc_{data_cut}": acc.detach().cpu().item()}

    def get_early_stopper(self) -> EarlyStopper:
        return EarlyStopper(patience=self.model_cfg.early_stop_patience)

    def compute_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.n_classes is None:
            return nn.functional.mse_loss(input=y_hat.squeeze(2), target=y)
        else:
            return nn.functional.cross_entropy(input=y_hat.flatten(end_dim=-2), target=y.flatten()).mean(dim=0)

    def train(
        self,
        model: nn.Module,
        al_step: int,
        train_loader: Optional[DataLoader],
        val_loader: Optional[DataLoader] = None,
        full_monitoring: bool = False,
    ) -> Tuple[nn.Module, Dict[str, List[float]]]:
        """
        This function trains a model on the train_loader and returns the best model according to the validation loss.
        It is agnostic to the model architecture and the task at hand.
        """
        if train_loader is None:
            return model, {}

        best_model = []
        running_log = defaultdict(list)

        # optimizer

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.model_cfg.lr, weight_decay=self.model_cfg.l2_reg)
        early_stopper = self.get_early_stopper()

        # training loop

        for epoch in range(self.model_cfg.train_epochs):

            model.train()
            train_loss = 0.0
            batch_loading_time = 0.0
            batch_processing_time = 0.0
            start = time.time()

            for X, y, _ in train_loader:
                batch_loading_time += time.time() - start
                start = time.time()

                X = X.to(self.device)
                y = y.to(self.device)
                if self.use_ensmbl:
                    X = X.unsqueeze(0).repeat(self.model_cfg.num_ensmbl_members, 1, 1)
                    y = y.unsqueeze(0).repeat(self.model_cfg.num_ensmbl_members, 1)

                optimizer.zero_grad()
                logit = model(X)
                loss = self.compute_loss(logit, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.model_cfg.grad_clip_norm)
                optimizer.step()
                train_loss += float(loss.item())
                batch_processing_time += time.time() - start
                start = time.time()

            train_loss_avg = train_loss / len(train_loader)
            batch_loading_time = batch_loading_time / len(train_loader)
            batch_processing_time = batch_processing_time / len(train_loader)

            # validation loop

            if val_loader is not None:
                model.eval()
                valid_loss = 0.0
                for X, y, _ in val_loader:
                    X = X.to(self.device)
                    y = y.to(self.device)
                    if self.use_ensmbl:
                        X = X.unsqueeze(0).repeat(self.model_cfg.num_ensmbl_members, 1, 1)
                        y = y.unsqueeze(0).repeat(self.model_cfg.num_ensmbl_members, 1)

                    logit = model(X)
                    loss = self.compute_loss(logit, y)
                    valid_loss += float(loss.item())

                valid_loss_avg = valid_loss / len(val_loader)
                self.early_stop_mask = early_stopper.add_observation(model, valid_loss_avg)

            # logging

            g_norm = {f"grad-norm-{name}": 0.0 for name, _ in model.named_parameters()}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if "batch" not in name:
                        g_norm[f"grad-norm-{name}"] = float(torch.norm(param.grad).item())

            running_log["train-loss"].append(float(train_loss_avg))
            running_log["validation-loss"].append(float(valid_loss_avg))
            running_log["first-layer-grad-norm"].append(float(list(g_norm.values())[0]))
            running_log["last-layer-grad-norm"].append(float(list(g_norm.values())[-2]))
            running_log["epoch"].append(epoch)

            if full_monitoring:
                with torch.no_grad():
                    train_probs, train_ys, _ = self.predict(model, train_loader)
                    train_metrics = self.compute_metrics(train_probs, train_ys, "inloop_train")
                    for k, v in train_metrics.items():
                        running_log[k].append(v)

                    valid_probs, valid_ys, _ = self.predict(model, val_loader)
                    valid_metrics = self.compute_metrics(valid_probs, valid_ys, "inloop_valid")
                    for k, v in valid_metrics.items():
                        running_log[k].append(v)

            if wandb.run is not None:
                wandb.log({k: v[-1] for k, v in running_log.items()})

            if early_stopper.check_condition():
                best_model = early_stopper.best_model
                break

        if epoch + 1 == self.model_cfg.train_epochs and not early_stopper.check_condition():
            best_model = deepcopy(model)

        return best_model, running_log

    def predict(
        self, model: nn.Module, dataloader: DataLoader, mc_dropout: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This function takes a model and a dataloader and returns the class probabilities and the true labels.
        It assumes a classification task, with the logits dimension being the last one.
        It is otherwise agnostic to the model architecture and the task at hand.
        """
        all_x_idxs = []

        model.eval()
        if mc_dropout:
            model.train()
        batch_preds = []
        batch_x_idxs = []
        batch_y = []
        for X, y, x_idx in dataloader:
            X = X.to(self.device)
            y = y.to(self.device)
            if self.use_ensmbl:
                X = X.unsqueeze(0).repeat(self.model_cfg.num_ensmbl_members, 1, 1)
                y = y.unsqueeze(0).repeat(self.model_cfg.num_ensmbl_members, 1)

            logits = model(X)

            if self.n_classes is not None:
                preds = torch.nn.functional.softmax(logits, dim=-1)
            else:
                preds = logits.squeeze(-1).movedim(0, 1)
                y = y.movedim(0, 1)

            batch_y.append(y)
            batch_x_idxs.append(x_idx)
            batch_preds.append(preds.detach())

        all_y = torch.cat(batch_y, dim=0).detach()
        all_x_idxs = torch.cat(batch_x_idxs, dim=0).detach()
        all_preds = torch.cat(batch_preds, dim=0).detach()

        return all_preds, all_y, all_x_idxs
