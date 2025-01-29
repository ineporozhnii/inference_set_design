import json
import os
import pickle
import random
import time
from collections import defaultdict
from dataclasses import fields
from pathlib import Path
from shutil import rmtree
from typing import Optional

import numpy as np
import torch
import wandb

from inference_set_design.agents.active_agent import ActiveAgent
from inference_set_design.agents.bald_agent import BALDAgent
from inference_set_design.agents.diversity_agents import (
    DiveristyCosineAgent,
    DiveristyEuclideanAgent,
    DiveristyTanimotoAgent,
)
from inference_set_design.agents.grid_agent import GridAgent
from inference_set_design.agents.random_agent import RandomAgent
from inference_set_design.agents.sa_agent import SyntheticAccessibilityAgent
from inference_set_design.config import Config, check_config
from inference_set_design.tasks.corrupted_mnist import CorruptedMNISTAcquisition
from inference_set_design.tasks.mol3d import Mol3DCompoundAcquisition
from inference_set_design.tasks.qm9 import QM9CompoundAcquisition
from inference_set_design.utils.misc import (
    create_logger,
    hms_time_fmt,
    nested_dict_to_flat_dict,
)


class ActiveLearningEnvironment:
    def __init__(self, cfg: Optional[Config] = None):
        self.setup_config(cfg)
        self.setup_device()
        self.seed_everything()

        self.log_path = Path(self.cfg.log_path)
        self.create_log_path()
        self.logger = create_logger(logfile=self.log_path / "logger.out")
        self.save_config()

        self.setup_task()
        self.setup_agent()

        self.metrics = defaultdict(list)
        self.additional_info = defaultdict(list)  # not logged in wandb

    def setup_config(self, cfg: Config):
        # initialise default config and merge user-provided config in
        self.cfg = Config()
        if cfg is not None:
            self.cfg = self.cfg.merge(cfg)
        # set all task fields to None except for selected task
        for field in fields(self.cfg.task_cfg):
            if field.name != self.cfg.task_name:
                self.cfg.task_cfg.__setattr__(field.name, None)

    def setup_device(self):
        if self.cfg.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA not available, falling back to CPU.")
        self.device = torch.device(self.cfg.device)

        # updating config with hardware info
        self.cfg.device = str(self.device)
        self.cfg.hostname = os.uname().nodename

    def seed_everything(self):
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)

    def create_log_path(self):
        if not self.log_path.exists():
            os.makedirs(self.log_path, exist_ok=True)
        else:
            if self.cfg.overwrite_run:
                rmtree(self.log_path)
                os.makedirs(self.log_path, exist_ok=True)
            else:
                raise ValueError(f"Log path already exists: {self.log_path}")

    def save_config(self):
        self.cfg = check_config(self.cfg, self.logger)
        with open(self.log_path / "config.json", "w") as f:
            json.dump(nested_dict_to_flat_dict(self.cfg.to_dict(), sep="."), f, indent=4)

        if wandb.run is not None:
            wandb.run.config.update(self.cfg, allow_val_change=True)
            wandb.run.save()

        self.logger.info(f"Config:\n{'-'*10}\n{self.cfg}{'-'*10}")

    def setup_task(self):
        assert self.cfg.task_name in [
            field.name for field in fields(self.cfg.task_cfg)
        ], f"Task {self.cfg.task_name} should be defined in TaskConfig's fields."
        generic_kwargs = {
            "model_cfg": self.cfg.model_cfg,
            "log_path": self.log_path,
            "logger": self.logger,
            "device": self.device,
            "reveal_all": self.cfg.acquisition_batch_size is None,
        }
        if self.cfg.task_name == "qm9":
            TaskClass = QM9CompoundAcquisition
            task_specific_kwargs = {
                "task_cfg": self.cfg.task_cfg.qm9,
            }
        elif self.cfg.task_name == "mol3d":
            TaskClass = Mol3DCompoundAcquisition
            task_specific_kwargs = {
                "task_cfg": self.cfg.task_cfg.mol3d,
            }
        elif self.cfg.task_name == "corrupted_mnist":
            TaskClass = CorruptedMNISTAcquisition
            task_specific_kwargs = {
                "task_cfg": self.cfg.task_cfg.corrupted_mnist,
            }
        else:
            raise NotImplementedError(f"Task {self.cfg.task_name} not implemented.")

        self.task = TaskClass(**{**generic_kwargs, **task_specific_kwargs})

    def setup_agent(self):
        if self.cfg.agent_name == "random":
            AgentClass = RandomAgent
        elif self.cfg.agent_name == "grid":
            AgentClass = GridAgent
        elif self.cfg.agent_name == "active":
            AgentClass = ActiveAgent
        elif self.cfg.agent_name == "bald":
            AgentClass = BALDAgent
        elif self.cfg.agent_name == "diversity_tanimoto":
            AgentClass = DiveristyTanimotoAgent
            self.agent = AgentClass(
                emb_name=self.task.cfg.emb_name,
                data_path=self.task.cfg.data_path,
                n_explorable_cmpds=self.task.cfg.n_explorable_cmpds,
                agent_cfg=self.cfg.agent_cfg,
                task=self.task,
                acquisition_batch_size=self.cfg.acquisition_batch_size,
                device=self.device,
                log_path=self.log_path,
            )
            return
        elif self.cfg.agent_name == "diversity_cosine":
            AgentClass = DiveristyCosineAgent
            self.agent = AgentClass(
                emb_name=self.task.cfg.emb_name,
                data_path=self.task.cfg.data_path,
                n_explorable_cmpds=self.task.cfg.n_explorable_cmpds,
                agent_cfg=self.cfg.agent_cfg,
                task=self.task,
                acquisition_batch_size=self.cfg.acquisition_batch_size,
                device=self.device,
                log_path=self.log_path,
            )
            return
        elif self.cfg.agent_name == "diversity_euclidean":
            AgentClass = DiveristyEuclideanAgent
            self.agent = AgentClass(
                emb_name=self.task.cfg.emb_name,
                data_path=self.task.cfg.data_path,
                n_explorable_cmpds=self.task.cfg.n_explorable_cmpds,
                agent_cfg=self.cfg.agent_cfg,
                task=self.task,
                acquisition_batch_size=self.cfg.acquisition_batch_size,
                device=self.device,
                log_path=self.log_path,
            )
            return
        elif self.cfg.agent_name == "sa":
            AgentClass = SyntheticAccessibilityAgent
            self.agent = AgentClass(
                data_path=self.task.cfg.data_path,
                n_explorable_cmpds=self.task.cfg.n_explorable_cmpds,
                agent_cfg=self.cfg.agent_cfg,
                task=self.task,
                acquisition_batch_size=self.cfg.acquisition_batch_size,
                device=self.device,
                log_path=self.log_path,
            )
            return
        else:
            raise ValueError(f"Agent type {self.cfg.agent_name} not recognized.")

        self.agent = AgentClass(
            agent_cfg=self.cfg.agent_cfg,
            task=self.task,
            acquisition_batch_size=self.cfg.acquisition_batch_size,
            device=self.device,
            log_path=self.log_path,
        )

    def checkpoint_logs(self):
        with open(self.log_path / "metrics.json", "w") as f:
            json.dump(self.metrics, f)

        if wandb.run is not None:
            wandb.log({k: v[-1] for k, v in self.metrics.items()})

        with open(self.log_path / "additional_info.pkl", "wb") as f:
            pickle.dump(self.additional_info, f)

    def run_active_learning_loop(self):
        batch_i = 1
        initial_action_space_size = np.argwhere(self.task.acquisition_mask).flatten().shape[0]

        while np.any(self.task.acquisition_mask) or not self.cfg.acquisition_batch_size:
            self.logger.info(f"Running batch {batch_i}...")
            start = time.time()

            acquisition_idxs, outer_loop_metrics, acquisition_metrics = self.agent.select_acquisition_batch(
                self.task.acquisition_mask
            )

            for k, v in {**outer_loop_metrics, **acquisition_metrics}.items():
                self.metrics[k].append(v)

            self.metrics["data_explored"].append(
                (initial_action_space_size - np.argwhere(self.task.acquisition_mask).flatten().shape[0])
                / initial_action_space_size
                if initial_action_space_size
                else 0
            )

            self.additional_info["acquisition_idxs"].append(acquisition_idxs)
            self.checkpoint_logs()

            self.task.label_batch(acquisition_idxs)

            if not self.cfg.acquisition_batch_size:
                # one-off experiment: we just run one batch and break
                break

            self.logger.info(f"Batch {batch_i} complete. Time elapsed: {hms_time_fmt(time.time() - start)}.")
            batch_i += 1

        self.logger.info("Active learning loop complete.")
