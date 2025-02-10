import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from inference_set_design.agents.config import ActiveAgentConfig
from inference_set_design.tasks.base_task import BaseTask
from inference_set_design.utils.misc import get_al_step_log_path


class BaseAgent(ABC):
    def __init__(
        self,
        agent_cfg: ActiveAgentConfig,
        task: BaseTask,
        acquisition_batch_size: int,
        device: str,
        log_path: Path,
    ):
        self.cfg = agent_cfg
        self.task = task
        self.acquisition_batch_size = acquisition_batch_size

        self.device = device
        self.log_path = log_path

        self.al_step = 0
        self.inner_loop_metrics = {}

    def get_explorable_predictions(self, model: nn.Module, dataloader: DataLoader):
        all_class_probs, all_labels, all_x_idxs = self.task.predict(
            model=model,
            dataloader=dataloader,
        )
        return all_class_probs, all_labels, all_x_idxs

    def get_metrics(self, model: nn.Module, loader: DataLoader, data_cut: str):
        if loader:
            class_probs, labels, _ = self.task.predict(model=model, dataloader=loader)
            metrics = self.task.compute_metrics(class_probs, labels, data_cut)
        else:
            metrics = {}
        return metrics

    @abstractmethod
    def acquisition_function(self, available_indices: np.ndarray, acquisition_scores: dict) -> np.ndarray:
        raise NotImplementedError("acquisition_function() needs to be implemented by the child class")

    def select_acquisition_batch(self, acquisition_mask: np.ndarray):

        # train on revealed data

        trained_model, train_metrics = self.task.train(
            model=self.task.initialise_model(),
            al_step=self.al_step,
            train_loader=self.task.get_train_dataloader(acquisition_mask),
            val_loader=self.task.get_valid_dataloader(),
            full_monitoring=self.cfg.full_inner_loop_monitoring,
        )
        self.inner_loop_metrics[f"al-step-{self.al_step}"] = train_metrics
        with open(self.log_path / "inner_loop_metrics.json", "w") as f:
            json.dump(self.inner_loop_metrics, f)

        # inference on hidden data

        with torch.no_grad():
            all_class_probs, all_labels, all_x_idxs = self.get_explorable_predictions(
                model=trained_model,
                dataloader=self.task.get_explorable_dataloader(),
            )
            metrics_explorable = self.task.compute_metrics(all_class_probs, all_labels, data_cut="explorable")

        al_step_path = get_al_step_log_path(self.log_path, self.al_step)
        al_step_path.mkdir(parents=True, exist_ok=True)

        if self.cfg.log_explorable_preds:
            np.save(al_step_path / "explorable_preds.npy", all_class_probs.cpu().numpy())
            np.save(al_step_path / "explorable_x_idxs.npy", all_x_idxs.cpu().numpy())

        if self.cfg.save_model:
            self.task.save_model(path=al_step_path, model=trained_model)

        metrics_train = self.get_metrics(
            trained_model, loader=self.task.get_train_dataloader(acquisition_mask), data_cut="train"
        )
        metrics_hidden = self.get_metrics(
            trained_model, loader=self.task.get_hidden_dataloader(acquisition_mask), data_cut="hidden"
        )
        metrics_test = self.get_metrics(trained_model, loader=self.task.get_test_dataloader(), data_cut="test")

        # get task-specific inference metrics

        acquisition_metrics, acquisition_scores = self.task.get_acquisition_scores(
            all_class_probs.cpu().numpy(), all_x_idxs.cpu().numpy()
        )

        # select samples to be labeled

        if self.acquisition_batch_size:
            available_indices = np.argwhere(acquisition_mask).flatten()

            if len(available_indices) > self.acquisition_batch_size:
                acquisition_batch = self.acquisition_function(available_indices, acquisition_scores)
                if len(acquisition_batch) > self.acquisition_batch_size:
                    acquisition_batch = acquisition_batch[: self.acquisition_batch_size]
            else:
                acquisition_batch = available_indices
        else:
            acquisition_batch = []

        metrics_acquisition_batch = self.get_metrics(
            trained_model, loader=self.task.get_batch_dataloader(acquisition_batch), data_cut="acquisition_batch"
        )

        np.save(al_step_path / "acquisition_batch.npy", acquisition_batch)

        metrics_system = {}
        for metric_name, metric_val in metrics_hidden.items():
            # Classification task
            if self.task.n_classes is not None:
                metrics_system[metric_name.replace("hidden", "system")] = (
                    metric_val * len(np.argwhere(acquisition_mask == 1).flatten())
                    + 1 * len(np.argwhere(acquisition_mask == 0).flatten())
                ) / len(acquisition_mask.flatten())
            # Regression task
            else:
                metrics_system[metric_name.replace("hidden", "system")] = (
                    metric_val * len(np.argwhere(acquisition_mask == 1).flatten())
                    + 0 * len(np.argwhere(acquisition_mask == 0).flatten())
                ) / len(acquisition_mask.flatten())

        outer_loop_metrics = {
            "al_step": self.al_step,
            **metrics_train,
            **metrics_hidden,
            **metrics_explorable,
            **metrics_test,
            **metrics_acquisition_batch,
            **metrics_system,
        }

        self.al_step += 1

        return acquisition_batch, outer_loop_metrics, acquisition_metrics
