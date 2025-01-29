from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import git
import numpy as np
import torch
from omegaconf import MISSING

from inference_set_design.agents.config import ActiveAgentConfig
from inference_set_design.models.config import ModelConfig
from inference_set_design.tasks.config import TaskConfig
from inference_set_design.utils.misc import StrictDataClass, flat_dict_to_nested_dict


@dataclass(repr=False)
class Config(StrictDataClass):
    # System parameters ---
    hostname: str = MISSING
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    git_hash: str = MISSING
    try:
        git_hash = git.Repo(os.path.realpath(__file__), search_parent_directories=True).head.object.hexsha[:7]
    except git.InvalidGitRepositoryError:
        pass

    # Experiment parameters ---
    agent_name: str = "random"
    agent_cfg: ActiveAgentConfig = field(default_factory=ActiveAgentConfig)
    task_name: str = "rxrx3"
    task_cfg: TaskConfig = field(default_factory=TaskConfig)
    model_cfg: ModelConfig = field(default_factory=ModelConfig)

    # Logging parameters ---
    seed: int = 42
    desc: str = "default configuration"
    log_path: str = "./runs/debug_run"
    overwrite_run: bool = False
    acquisition_batch_size: Optional[int] = 10


def check_config(cfg: Config, logger: logging.Logger):
    assert cfg.agent_name in [
        "sa",
        "bald",
        "random",
        "grid",
        "active",
        "diversity_tanimoto",
        "diversity_cosine",
        "diversity_euclidean",
    ], f"config.agent_type not supported: {cfg.agent_name}"
    assert cfg.agent_cfg.acquisition_strategy in [
        "greedy"
    ], f"config.acquisition_strategy not supported: {cfg.agent_cfg.acquisition_strategy}"
    assert np.sum(cfg.agent_cfg.acq_weights) == 1.0, "acq_weights must sum to 1.0"

    if cfg.task_cfg.qm9 is not None:
        if cfg.task_cfg.qm9.n_explorable_cmpds is not None:
            assert cfg.task_cfg.qm9.n_init_train_cmpds < cfg.task_cfg.qm9.n_explorable_cmpds, (
                f"n_train_cmpds ({cfg.task_cfg.qm9.n_init_train_cmpds})"
                f" > n_explorable_cmpds ({cfg.task_cfg.qm9.n_explorable_cmpds})"
            )
    assert cfg.model_cfg.model_name in [
        "MLP",
        "ResMLP",
        "MultiTaskMLP",
    ], f"model_name not supported: {cfg.model_cfg.model_name}"
    return cfg
