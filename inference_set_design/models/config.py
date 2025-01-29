from dataclasses import dataclass
from typing import Optional

from inference_set_design.utils.misc import StrictDataClass


@dataclass(repr=False)
class ModelConfig(StrictDataClass):
    model_name: str = "ResMLP"
    lr: float = 0.001
    l2_reg: float = 0.0
    grad_clip_norm: float = 1.0
    dropout: float = 0.1
    num_ensmbl_members: Optional[int] = None
    train_epochs: int = 1000
    train_batch_size: int = 1024
    early_stop_patience: Optional[int] = 50
    n_hidden_layers: int = 2
    hidden_size: int = 512
    skip_connections: bool = False

    # Multi-task config
    trunk_hidden_size: int = 512
    n_trunk_res_block: int = 2
    task_hidden_size: int = 512
    n_task_layers: int = 2
