from __future__ import annotations

import logging
import os.path
import sys
from copy import deepcopy
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np
import omegaconf
import torch
import torch.nn as nn
from omegaconf import MISSING, OmegaConf


def create_logger(name="logger", loglevel=logging.INFO, logfile=None, streamHandle=True):
    logger = logging.getLogger(name)
    logger.setLevel(loglevel)
    while len([logger.removeHandler(i) for i in logger.handlers]):
        pass  # Remove all handlers (only useful when debugging)
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - {} - %(message)s".format(name),
        datefmt="%d/%m/%Y %H:%M:%S",
    )

    handlers = []
    if logfile is not None:
        handlers.append(logging.FileHandler(logfile, mode="a"))
    if streamHandle:
        handlers.append(logging.StreamHandler(stream=sys.stdout))

    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def hms_time_fmt(time_in_seconds: float) -> str:
    h = int(time_in_seconds // 3600)
    m = int((time_in_seconds % 3600) // 60)
    s = int(time_in_seconds % 60)
    return f"{h:d}h{m:02d}m{s:02d}s"


class EarlyStopper:
    def __init__(self, patience: int = 5):
        self.patience = patience
        self.counter = 0
        self.min_validation_loss = float("inf")
        self.best_model = None

    def add_observation(self, model: nn.Module, validation_loss: float):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.best_model = deepcopy(model)
        else:
            self.counter += 1

    def check_condition(self):
        if self.patience is None:
            return False
        elif self.counter >= self.patience:
            assert self.best_model is not None, "Early Stopping should not be triggered if no model has been saved."
            return True
        else:
            return False


class SimpleDataloader:
    def __init__(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
        batch_size: int,
        shuffle: bool = False,
    ):
        self.xs = xs
        self.ys = ys
        self.dataset_len = len(xs)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        if len(self.xs) % self.batch_size == 0:
            return len(self.xs) // self.batch_size
        else:
            return (len(self.xs) // self.batch_size) + 1

    def __iter__(self):
        indices = np.arange(self.dataset_len)
        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            x_idx = torch.tensor(batch_indices, dtype=torch.long)
            X = self.xs[batch_indices]
            y = self.ys[batch_indices]
            yield X, y, x_idx


class StrictDataClass:
    def __repr__(self) -> str:
        return OmegaConf.to_yaml(self)

    def merge(self, update_cfg: StrictDataClass) -> StrictDataClass:
        """Merges another (nested) dataclass instance into this one if value is not MISSING."""
        if update_cfg is None:
            return self
        for f in fields(self):
            update_value = getattr(update_cfg, f.name, MISSING)
            if update_value is not MISSING:
                if is_dataclass(getattr(self, f.name, None)):
                    nested_obj = getattr(self, f.name)
                    if isinstance(nested_obj, StrictDataClass):
                        nested_obj = nested_obj.merge(update_value)
                    else:
                        setattr(self, f.name, update_value)
                else:
                    setattr(self, f.name, update_value)

        return self

    def merge_dict(self, update_dict: dict) -> StrictDataClass:
        """Merges another (nested) dictionary into this (nested) dataclass if value is not MISSING."""
        if update_dict is None:
            return self
        for key in update_dict.keys():
            if key not in self.__annotations__:
                raise KeyError(f"Key '{key}' not in dataclass '{type(self).__name__}'.")
        for f in fields(self):
            if f.name in update_dict:
                update_value = update_dict[f.name]
                if is_dataclass(getattr(self, f.name, None)):
                    nested_obj = getattr(self, f.name)
                    if isinstance(nested_obj, StrictDataClass):
                        nested_obj = nested_obj.merge_dict(update_value)
                    else:
                        setattr(self, f.name, update_value)
                else:
                    setattr(self, f.name, update_value)

        return self

    def __setattr__(self, name, value):
        if hasattr(self, name) or name in self.__annotations__:
            super().__setattr__(name, value)
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'."
                f" Attributes can only be defined in the class definition."
            )

    def to_dict(self):
        """Returns a (nested) dictionary representation of the dataclass."""
        dict_rep = {}
        for f in fields(self):
            if is_dataclass(f.default_factory) and hasattr(getattr(self, f.name), "to_dict"):
                dict_rep[f.name] = getattr(self, f.name).to_dict()
            else:
                dict_rep[f.name] = getattr(self, f.name)
        return dict_rep

    @classmethod
    def empty(cls):
        """
        Creates a new instance of the (nested) dataclass with all fields set to MISSING.
        This allows creating a configuration template to be filled by the user.
        """
        cfg = cls()
        for f in fields(cls):
            if is_dataclass(f.default_factory):
                setattr(cfg, f.name, f.default_factory.empty())
            else:
                setattr(cfg, f.name, MISSING)
        return cfg


def flat_dict_to_nested_dict(linear_dict, sep="."):
    """
    Converts a dictionary with linear path-based keys to a nested dictionary, recursively.

    Args:
    - linear_dict (dict): The input dictionary with linear path-based keys.
    - nested_char (str): The character used to denote nesting in the keys.

    Returns:
    - dict: The nested dictionary.

    Example:
    >>> linear_dict_to_nested_dict({"a.b.c": 1})
    {'a': {'b': {'c': 1}}}
    """

    def merge_dicts(a, b):
        for key, value in b.items():
            if key in a and isinstance(a[key], dict) and isinstance(value, dict):
                merge_dicts(a[key], value)
            else:
                a[key] = value

    nested_dict = {}
    for key, value in linear_dict.items():
        if sep in key:
            key1, key2 = key.split(sep, 1)
            if key1 not in nested_dict:
                nested_dict[key1] = {}
            merge_dicts(nested_dict[key1], flat_dict_to_nested_dict({key2: value}))
        else:
            nested_dict[key] = value

    return nested_dict


def nested_dict_to_flat_dict(nested_dict, sep="."):
    """
    Converts a nested dictionary to a dictionary with linear path-based keys, recursively.

    Args:
    - nested_dict (dict): The input dictionary with nested keys.
    - nested_char (str): The character used to denote nesting in the keys.

    Returns:
    - dict: The dictionary with linear path-based keys.

    Example:
    >>> nested_dict_to_linear_dict({'a': {'b': {'c': 1}}})
    {'a.b.c': 1}
    """
    if not isinstance(nested_dict, dict):
        return nested_dict

    linear_dict = {}
    for key, value in nested_dict.items():
        if not isinstance(value, dict):
            linear_dict[key] = value
        else:
            flat_dict = nested_dict_to_flat_dict(value)
            for flat_key, flat_value in flat_dict.items():
                linear_dict[f"{key}{sep}{flat_key}"] = flat_value
    return linear_dict


def get_al_step_log_path(log_path: Path, al_step: int) -> Path:
    return log_path / f"al_step_{al_step}"
