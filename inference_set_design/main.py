from inference_set_design.config import Config
from inference_set_design.env import ActiveLearningEnvironment

if __name__ == "__main__":
    cfg = Config.empty()
    cfg.overwrite_run = True
    cfg.agent_name = "active"
    cfg.agent_cfg.acq_weights = (0.8, 0.2)
    cfg.model_cfg.train_epochs = 3
    cfg.model_cfg.num_ensmbl_members = None

    TASK = "rxrx3"    
    cfg.task_name = TASK

    if TASK == "corrupted_mnist":
        cfg.acquisition_batch_size = 1000
        cfg.model_cfg.model_name = "ResMLP"
        cfg.agent_cfg.log_explorable_preds = False
        cfg.agent_cfg.save_model = True

    elif TASK == "qm9":
        cfg.acquisition_batch_size = 1000
        cfg.model_cfg.model_name = "ResMLP"
        cfg.agent_cfg.log_explorable_preds = False
        cfg.task_cfg.qm9.n_explorable_cmpds = 10_000
        cfg.task_cfg.qm9.n_init_train_cmpds = 1000

    elif TASK == "mol3d":
        cfg.acquisition_batch_size = 1000
        cfg.model_cfg.model_name = "ResMLP"
        cfg.model_cfg.num_ensmbl_members = 5
        cfg.model_cfg.train_batch_size = 1024
        cfg.agent_cfg.log_explorable_preds = False
        cfg.agent_cfg.acq_criteria = ("std", "random")
        cfg.task_cfg.mol3d.n_explorable_cmpds = 10_000
        cfg.task_cfg.mol3d.data_path = (
            "./mol3d_data/splits"
        )
    
    elif TASK == "rxrx3":
        cfg.acquisition_batch_size = 25
        cfg.model_cfg.train_batch_size = 25
        cfg.model_cfg.model_name = "MultiTaskMLP"
        cfg.agent_cfg.log_explorable_preds = False

    env = ActiveLearningEnvironment(cfg)
    env.run_active_learning_loop()
