import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import RDConfig
from tqdm import tqdm

from inference_set_design.agents.base_agent import BaseAgent
from inference_set_design.agents.config import ActiveAgentConfig
from inference_set_design.tasks.base_task import BaseTask

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # type: ignore # noqa: E402


class SyntheticAccessibilityAgent(BaseAgent):
    def __init__(
        self,
        data_path: str,
        n_explorable_cmpds: int,
        agent_cfg: ActiveAgentConfig,
        task: BaseTask,
        acquisition_batch_size: int,
        device: str,
        log_path: Path,
    ):
        super().__init__(
            agent_cfg=agent_cfg,
            task=task,
            acquisition_batch_size=acquisition_batch_size,
            device=device,
            log_path=log_path,
        )

        # Load explorable data
        explr_df = pd.read_parquet(Path(data_path) / "explorable.parquet")

        # Reduce the number of compounds to use
        if n_explorable_cmpds is not None:
            assert n_explorable_cmpds <= explr_df.shape[0]
        self.n_explr_cmpds = explr_df.shape[0] if n_explorable_cmpds is None else n_explorable_cmpds
        explr_df = explr_df.iloc[: self.n_explr_cmpds]

        # Extract smiles from dataframe
        smiles = explr_df["smiles"].to_list()
        molecules = [Chem.MolFromSmiles(s) for s in tqdm(smiles, "Processing SMILEs ...")]
        self.sa_scores = [sascorer.calculateScore(mol) for mol in tqdm(molecules, "Computing SA scores ...")]

    def acquisition_function(self, available_indices: np.ndarray, acquisition_scores: dict):
        # Use SA scores as acquisition scores
        acquisition_scores = self.sa_scores
        # Sort samples based on acquisition scores and select a batch
        idx_score = [(idx, acquisition_scores[idx]) for idx in available_indices]
        idx_score = sorted(idx_score, key=lambda x: x[1], reverse=True)
        acquisition_batch = np.array([idx for idx, _ in idx_score[: self.acquisition_batch_size]])
        return acquisition_batch
