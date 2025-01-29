from dataclasses import dataclass
from typing import Tuple

from inference_set_design.utils.misc import StrictDataClass


@dataclass(repr=False)
class ActiveAgentConfig(StrictDataClass):
    acquisition_strategy: str = "greedy"
    # the list of acquisition scores to consider
    # and their relative batch composition weights
    acq_criteria: Tuple[str] = ("uncertainty", "random")
    acq_weights: Tuple[float] = (1.0, 0.0)
    # whether to save the predictions of the explorable set to disk
    log_explorable_preds: bool = False
    # whether to save the model at each active learning step
    save_model: bool = False
    # whether to compute all metrics (or just losses) for the inner loop
    full_inner_loop_monitoring: bool = False
    # number of mc-dropout iterations for BALD agent
    mc_iterations: int = 5
