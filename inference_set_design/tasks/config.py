from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import inference_set_design
from inference_set_design.utils.misc import StrictDataClass


@dataclass(repr=False)
class QM9Config(StrictDataClass):
    n_explorable_cmpds: Optional[int] = None
    n_init_train_cmpds: int = 500
    emb_name: str = "ecfp4"
    label_name: str = "gap_binary"
    data_path: str = "data/QM9"


@dataclass(repr=False)
class Mol3DConfig(StrictDataClass):
    n_explorable_cmpds: Optional[int] = None
    n_init_train_cmpds: int = 500
    emb_name: str = "ecfp4"
    label_name: str = "homolumogap"
    data_path: str = "data/Mol3D"


@dataclass(repr=False)
class CorruptedMNIST(StrictDataClass):
    n_init_train_imgs: int = 0
    corruption_type: Optional[str] = "bottom"
    num_data_workers: Optional[int] = 16
    shuffle_datasets: bool = False
    data_path: str = "data/MNIST"


@dataclass(repr=False)
class RXRX3Config(StrictDataClass):
    n_explorable_cmpds: Optional[int] = None
    n_init_train_cmpds: int = 0
    emb_name: str = "molgps_fps"
    data_path: str = "data/RxRx3"
    use_class_balancing_weights: bool = True


@dataclass(repr=False)
class TaskConfig(StrictDataClass):
    qm9: Optional[QM9Config] = field(default_factory=QM9Config)
    mol3d: Optional[Mol3DConfig] = field(default_factory=Mol3DConfig)
    corrupted_mnist: Optional[CorruptedMNIST] = field(default_factory=CorruptedMNIST)
    rxrx3: Optional[RXRX3Config] = field(default_factory=RXRX3Config)
